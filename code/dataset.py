import json

import numpy as np
import pandas as pd

import utils


# TODO: what happens if I want to add new batches over time? Need a way to append to the current iterator.
class Dataset:
    def __init__(self, df, domain,
                 id_col, insertion_time_col, deletion_time_col, time_interval,
                 hist_repr_type="ohe"):
        """
        Wrapper for a dataset.

        :param df: pandas dataframe to wrap.
        :param domain: dict of feature names -> possible values each feature can take.
        :param id_col: Column name for the record ID.
        :param insertion_time_col: Column name for the insertion timestamp.
        :param deletion_time_col: Column name for the deletion timestamp.
        :param time_interval: pandas.DateOffset object that specifies
            the time interval to batch the dataset over.
        :param hist_repr_type: Encoding format for dataset. Supported formats: 'binarized', 'ohe' (default).
        """
        self.df = df
        self.domain = domain
        self.id_col = id_col
        self.insertion_time_col = insertion_time_col
        self.deletion_time_col = deletion_time_col
        self.time_interval = time_interval
        self.num_batches = self.create_batches()
        self.hist_repr_type = hist_repr_type
        self.encoded_df = None
        self.hist_repr_dim = None
        self.hist_repr_columns = None

    @staticmethod
    def load_from_path(path, domain_path,
                       id_col, insertion_time_col, deletion_time_col, time_interval,
                       hist_repr_type="ohe"):
        """
        Load dataset from CSV file and returned wrapped Dataset.

        :param path: CSV dataset to wrap.
        :param domain_path: JSON file of feature names -> possible values each feature can take.
        :param id_col: Column name for the record ID.
        :param insertion_time_col: Column name for the insertion timestamp.
        :param deletion_time_col: Column name for the deletion timestamp.
        :param time_interval: pandas.DateOffset object that specifies
            the time interval to batch the dataset over.
        :param hist_repr_type: Encoding format for dataset. Supported formats: 'binarized', 'ohe' (default).
        :return: A Dataset object wrapping the specified CSV dataset.
        """
        df = pd.read_csv(path)
        with open(domain_path, "r") as domain_file:
            domain = json.load(domain_file)
        return Dataset(df, domain, id_col, insertion_time_col, deletion_time_col, time_interval, hist_repr_type)

    def save_to_path(self, path):
        """
        Save dataset being wrapped to a CSV file.

        :param path: Path of CSV file where the dataframe will be saved.
        """
        self.df.to_csv(path, index=False)

    def select_rows_from_ids(self, ids):
        """
        Return pandas dataframe that only has rows with IDs == ids.

        :param ids: IDs to select from the wrapped dataset.
        :return: A pandas dataframe with selected rows.
        """
        return self.df[self.df[self.id_col].isin(ids)]

    def create_batches(self):
        """
        Create batches over insertion and deletion timestamps using the specified time interval.

        :return: Number of batches.
        """
        # convert insertion and deletion times to pandas timestamps
        self.df[self.insertion_time_col] = pd.to_datetime(self.df[self.insertion_time_col])
        self.df[self.deletion_time_col] = pd.to_datetime(self.df[self.deletion_time_col])

        # create bins according to the time interval
        insertion_times = self.df[self.insertion_time_col]
        deletion_times = self.df[self.deletion_time_col]
        start_time = min(insertion_times.min(), deletion_times.min()) - self.time_interval  # to cover start time
        end_time = max(insertion_times.max(), deletion_times.max()) + self.time_interval  # to cover end time
        batches = pd.date_range(start=start_time,
                                end=end_time,
                                freq=self.time_interval)

        # bucket insertion and deletion times according to time interval
        bucketed_insertions = pd.cut(insertion_times, bins=batches, include_lowest=True, labels=False)
        bucketed_deletions = pd.cut(deletion_times, bins=batches, include_lowest=True, labels=False)

        # assign batch numbers
        self.df["insertion_batch"] = np.array(bucketed_insertions, dtype=np.int64)
        self.df["deletion_batch"] = np.array(bucketed_deletions, dtype=np.int64)

        # set number of batches
        return max(bucketed_insertions.max(), bucketed_deletions.max())

    def get_batches(self):
        """
        Iterate over batches, returning the IDs of inserted and deleted rows in each batch.

        :return: Iterator that yields (insertion_ids, deletion_ids) per batch.
        """
        for batch_i in range(self.num_batches):
            insertion_ids = self.df[self.df["insertion_batch"] == batch_i][self.id_col].to_list()
            deletion_ids = self.df[self.df["deletion_batch"] == batch_i][self.id_col].to_list()
            yield insertion_ids, deletion_ids

    def get_domain(self):
        return self.domain

    def get_hist_repr_type(self):
        return self.hist_repr_type

    def get_hist_repr(self, ids):
        # encode dataset if it has not been encoded before
        if self.encoded_df is None:
            df_with_features_only = self.df.drop(columns=[self.id_col,
                                                          self.insertion_time_col, self.deletion_time_col,
                                                          "insertion_batch", "deletion_batch"])

            # encode dataset according to hist_repr_type
            encoded_df = pd.DataFrame({})
            if self.hist_repr_type == "binarized":
                encoded_df = utils.dataset_to_binarized(df_with_features_only, self.domain)
            elif self.hist_repr_type == "ohe":
                encoded_df = utils.dataset_to_ohe(df_with_features_only, self.domain)

            # save column names
            self.hist_repr_columns = encoded_df.columns

            # save encoded dataset
            self.encoded_df = encoded_df

        # select current ids from encoded dataset
        selected_ids = list(np.where(self.df[self.id_col].isin(ids))[0])
        reduced_encoded_df = self.encoded_df.iloc[selected_ids]

        # convert to numpy array
        reduced_encoded_df_arr = reduced_encoded_df.to_numpy()

        # compute histogram
        hist_repr_dim = self.get_hist_repr_dim()
        hist_repr = [0.0] * (2 ** hist_repr_dim)
        for row_idx in range(reduced_encoded_df_arr.shape[0]):  # iterate over all records in the dataset
            x = reduced_encoded_df_arr[row_idx]  # get current record

            # find bin corresponding to features for the record
            num = 0
            for dim_idx in range(hist_repr_dim):
                num += int(x[hist_repr_dim - dim_idx - 1]) * (2 ** dim_idx)

            # update count
            hist_repr[num] += 1.0
        hist_repr = np.array(hist_repr)
        hist_repr /= hist_repr.sum()  # normalize histogram

        return hist_repr

    def get_hist_repr_columns(self):
        return self.hist_repr_columns

    def get_hist_repr_dim(self):
        # calculate dim if it has not been calculated before
        if self.hist_repr_dim is None:
            # dim = sum of all feature domain sizes = columns in encoded dataset
            dim = 0
            if self.hist_repr_type == "ohe":
                for feature in self.domain.keys():
                    feature_domain = self.domain[feature]
                    if isinstance(feature_domain, int):
                        dim += feature_domain  # number of categories is already given as int
                    else:
                        dim += len(feature_domain)  # number of categories for discrete variable
            elif self.hist_repr_type == "binarized":
                for feature in self.domain.keys():
                    feature_domain = self.domain[feature]
                    if isinstance(feature_domain, str):
                        r = self.df[feature].max() - self.df[feature].min()  # range for continuous variable
                        dim += int(np.ceil(np.log2(r)))  # length of binarized representation
                    else:
                        num_categories = len(feature_domain)  # number of categories for discrete variable
                        dim += int(np.ceil(np.log2(num_categories)))  # length of binarized representation
            self.hist_repr_dim = dim
        return self.hist_repr_dim


# Testing
if __name__ == "__main__":
    # n_rows = 10000
    # create_fake_random_dataset(n_rows)
    #
    # time_int = pd.DateOffset(days=1)
    # time_int_str = "1day"
    # fake_dataset = Dataset.load_from_path(f"../data/fake_random_dataset_{n_rows}.csv",
    #                                       domain_path=f"../data/fake_random_dataset_{n_rows}_domain.json",
    #                                       id_col="Person ID",
    #                                       insertion_time_col="Insertion Time",
    #                                       deletion_time_col="Deletion Time",
    #                                       time_interval=time_int)
    # fake_dataset.save_to_path(f"../data/fake_random_dataset_{n_rows}_batched_{time_int_str}.csv")
    # for i, (ins_ids, del_ids) in enumerate(fake_dataset.get_batches()):
    #     print("Batch:", i)
    #     print("Insertions:", ins_ids)
    #     print("Deletions", del_ids)

    dataset_name = "adult_small"
    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    pmw_encoding_type = "binarized"
    dataset = Dataset.load_from_path(f"../data/{dataset_name}_{pmw_encoding_type}.csv",
                                     domain_path=f"../data/{dataset_name}_{pmw_encoding_type}_domain.json",
                                     id_col="Person ID",
                                     insertion_time_col="Insertion Time",
                                     deletion_time_col="Deletion Time",
                                     time_interval=time_int,
                                     hist_repr_type=pmw_encoding_type)
    dataset.save_to_path(f"../data/{dataset_name}_{pmw_encoding_type}_batched_{time_int_str}.csv")
    for i, (ins_ids, del_ids) in enumerate(dataset.get_batches()):
        print("Batch:", i)
        print("Insertions:", ins_ids)
        print("Deletions", del_ids)

    hist = dataset.get_hist_repr(ids=dataset.df[dataset.id_col])
    print("Dimension of the hist representation:", dataset.get_hist_repr_dim())
