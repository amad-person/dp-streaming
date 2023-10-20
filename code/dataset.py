import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def create_toy_dataset(path=None):
    person_ids = list(range(25))
    insertion_times = ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01',
                       '2023-01-02', '2023-01-02', '2023-01-02', '2023-01-02', '2023-01-02',
                       '2023-01-03', '2023-01-03', '2023-01-03', '2023-01-03', '2023-01-03',
                       '2023-01-04', '2023-01-04', '2023-01-04', '2023-01-04', '2023-01-04',
                       '2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05']
    deletion_times = ['2023-01-02', '2023-01-02', '2023-01-02', '2023-01-02', '2023-01-02',
                      '2023-01-03', '2023-01-03', '2023-01-03', '2023-01-03', '2023-01-03',
                      '2023-01-04', '2023-01-04', '2023-01-04', '2023-01-04', '2023-01-04',
                      '2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05',
                      '2023-01-06', '2023-01-06', '2023-01-06', '2023-01-06', '2023-01-06']

    # save to file
    data_dict = {
        "Person ID": person_ids,
        "Insertion Time": insertion_times,
        "Deletion Time": deletion_times
    }
    df = pd.DataFrame(data=data_dict)

    if path is None:
        path = "toy_dataset.csv"
    df.to_csv(path, index=False)


def create_fake_random_dataset(num_rows=10000, path=None):
    """
    Create fake dataset with "Person ID", "Insertion Time", "Deletion Time" cols.

    :param num_rows: Number of rows in fake dataset.
    :param path: Path of CSV file where the fake dataset will be saved.
    """
    person_ids = list(range(num_rows))

    # generate random insertion times for each person
    start_date = datetime(year=2023, month=1, day=1)
    end_date = datetime(year=2023, month=12, day=31)
    possible_timestamps = np.arange(start_date, end_date, dtype="datetime64[s]")
    insertion_times = np.random.choice(possible_timestamps, size=num_rows)

    # generate random deletion times for each person (deletion times should be after insertion times)
    time_deltas = np.array([timedelta(days=np.random.randint(1, 30)) for _ in range(num_rows)],
                           dtype="timedelta64[s]")
    deletion_times = insertion_times + time_deltas
    deletion_times = np.where(deletion_times > end_date, end_date, deletion_times)

    # save to file
    data_dict = {
        "Person ID": person_ids,
        "Insertion Time": insertion_times,
        "Deletion Time": deletion_times
    }
    df = pd.DataFrame(data=data_dict)

    if path is None:
        path = f"fake_random_dataset_{num_rows}.csv"
    df.to_csv(path, index=False)


def create_fake_ins_after_del_dataset(num_ins=2 ** 20, num_repeats=6, path=None):
    person_ids = list(range(num_ins * num_repeats))
    insertion_times, deletion_times = [], []
    start_date = datetime(year=2023, month=1, day=1)
    for r in range(num_repeats):
        insertion_times += [start_date] * num_ins
        start_date += timedelta(days=1)
        deletion_times += [start_date] * num_ins
        start_date += timedelta(days=1)

    # save to file
    data_dict = {
        "Person ID": person_ids,
        "Insertion Time": insertion_times,
        "Deletion Time": deletion_times
    }
    df = pd.DataFrame(data=data_dict)

    if path is None:
        path = f"fake_ins_after_del_dataset_{num_ins}_{num_repeats}.csv"
    df.to_csv(path, index=False)


# TODO: what happens if I want to add new batches over time? Need a way to append to the current iterator.
class Dataset:
    def __init__(self, df, id_col, insertion_time_col, deletion_time_col, time_interval):
        """
        Wrapper for a dataset.

        :param df: pandas dataframe to wrap.
        :param id_col: Column name for the record ID.
        :param insertion_time_col: Column name for the insertion timestamp.
        :param deletion_time_col: Column name for the deletion timestamp.
        :param time_interval: pandas.DateOffset object that specifies
            the time interval to batch the dataset over.
        """
        self.df = df
        self.id_col = id_col
        self.insertion_time_col = insertion_time_col
        self.deletion_time_col = deletion_time_col
        self.time_interval = time_interval
        self.num_batches = self.create_batches()

    @staticmethod
    def load_from_path(path, id_col, insertion_time_col, deletion_time_col, time_interval):
        """
        Load dataset from CSV file and returned wrapped Dataset.

        :param path: CSV dataset to wrap.
        :param id_col: Column name for the record ID.
        :param insertion_time_col: Column name for the insertion timestamp.
        :param deletion_time_col: Column name for the deletion timestamp.
        :param time_interval: pandas.DateOffset object that specifies
            the time interval to batch the dataset over.
        :return: A Dataset object wrapping the specified CSV dataset.
        """
        df = pd.read_csv(path)
        return Dataset(df, id_col, insertion_time_col, deletion_time_col, time_interval)

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


# Testing
if __name__ == "__main__":
    n_rows = 10000
    create_fake_random_dataset(n_rows)

    time_int = pd.DateOffset(days=1)
    time_int_str = "1day"
    fake_dataset = Dataset.load_from_path(f"fake_dataset_{n_rows}.csv",
                                          id_col="Person ID",
                                          insertion_time_col="Insertion Time",
                                          deletion_time_col="Deletion Time",
                                          time_interval=time_int)
    fake_dataset.save_to_path(f"fake_dataset_{n_rows}_batched_{time_int_str}.csv")
    for i, (ins_ids, del_ids) in enumerate(fake_dataset.get_batches()):
        print("Batch:", i)
        print("Insertions:", ins_ids)
        print("Deletions", del_ids)
