import numpy as np
import pandas as pd
import json
from data.preprocessor import create_fake_random_dataset


# TODO: what happens if I want to add new batches over time? Need a way to append to the current iterator.
class Dataset:
    def __init__(self, df, domain, id_col, insertion_time_col, deletion_time_col, time_interval):
        """
        Wrapper for a dataset.

        :param df: pandas dataframe to wrap.
        :param domain: dict of feature names -> possible values each feature can take.
        :param id_col: Column name for the record ID.
        :param insertion_time_col: Column name for the insertion timestamp.
        :param deletion_time_col: Column name for the deletion timestamp.
        :param time_interval: pandas.DateOffset object that specifies
            the time interval to batch the dataset over.
        """
        self.df = df
        self.domain = domain
        self.id_col = id_col
        self.insertion_time_col = insertion_time_col
        self.deletion_time_col = deletion_time_col
        self.time_interval = time_interval
        self.num_batches = self.create_batches()

    @staticmethod
    def load_from_path(path, domain_path, id_col, insertion_time_col, deletion_time_col, time_interval):
        """
        Load dataset from CSV file and returned wrapped Dataset.

        :param path: CSV dataset to wrap.
        :param domain_path: JSON file of feature names -> possible values each feature can take.
        :param id_col: Column name for the record ID.
        :param insertion_time_col: Column name for the insertion timestamp.
        :param deletion_time_col: Column name for the deletion timestamp.
        :param time_interval: pandas.DateOffset object that specifies
            the time interval to batch the dataset over.
        :return: A Dataset object wrapping the specified CSV dataset.
        """
        df = pd.read_csv(path)
        with open(domain_path, "r") as domain_file:
            domain = json.load(domain_file)
        return Dataset(df, domain, id_col, insertion_time_col, deletion_time_col, time_interval)

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

    def get_hist_repr(self, ids):
        reduced_df = self.select_rows_from_ids(ids)

        df_bins = []
        for feature in self.domain.keys():
            feature_domain = self.domain[feature]
            min_feature_value = 0
            if isinstance(feature_domain, int):
                max_feature_value = feature_domain - 1
            else:
                max_feature_value = len(feature_domain) - 1
            bins = list(range(max_feature_value - min_feature_value + 2))
            df_bins.append(bins)
        df_arr = reduced_df.drop(columns=[self.id_col, self.insertion_time_col, self.deletion_time_col,
                                 "insertion_batch", "deletion_batch"]).to_numpy()
        df_hist, df_hist_edges = np.histogramdd(df_arr, bins=df_bins)

        # test histogram representation
        query_for_df = ""
        for idx, feature in enumerate(self.domain.keys()):
            query_for_df += f"`{feature}` == 0"
            if idx != (len(self.domain.keys()) - 1):
                query_for_df += " & "
        assert df_hist.flatten()[0] == reduced_df.query(query_for_df).shape[0]

        return df_hist, df_hist_edges

    def get_synthetic_hist_shape(self):
        shape = []
        for feature in self.domain.keys():
            feature_domain = self.domain[feature]
            if isinstance(feature_domain, int):
                shape.append(feature_domain)
            else:
                shape.append(len(feature_domain))
        return shape


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

    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    adult_dataset = Dataset.load_from_path("../data/adult_reduced.csv",
                                           domain_path="../data/adult_reduced_domain.json",
                                           id_col="Person ID",
                                           insertion_time_col="Insertion Time",
                                           deletion_time_col="Deletion Time",
                                           time_interval=time_int)
    adult_dataset.save_to_path(f"../data/adult_reduced_batched_{time_int_str}.csv")
    for i, (ins_ids, del_ids) in enumerate(adult_dataset.get_batches()):
        print("Batch:", i)
        print("Insertions:", ins_ids)
        print("Deletions", del_ids)

    hist, edges = adult_dataset.get_hist_repr(ids=adult_dataset.df[adult_dataset.id_col])
