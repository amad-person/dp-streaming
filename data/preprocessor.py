import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from rdt import HyperTransformer
from rdt.transformers import OrderedLabelEncoder


def create_toy_dataset(path=None, domain_path=None):
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
        path = "./toy_dataset.csv"
    df.to_csv(path, index=False)

    domain_json = json.dumps({})
    if domain_path is None:
        domain_path = "./toy_dataset_domain.json"
    with open(domain_path, "w") as domain_file:
        domain_file.write(domain_json)


def create_fake_random_dataset(num_rows=10000, path=None, domain_path=None):
    """
    Create fake dataset with "Person ID", "Insertion Time", "Deletion Time" cols.

    :param num_rows: Number of rows in fake dataset.
    :param path: Path of CSV file where the fake dataset will be saved.
    :param domain_path: Path of JSON file with feature domain information.
    """
    person_ids = list(range(num_rows))

    # generate random insertion times for each person
    start_date = datetime(year=2023, month=1, day=1)
    end_date = datetime(year=2023, month=12, day=31)
    insertion_times, deletion_times = get_random_ins_and_del_times(num_rows=num_rows,
                                                                   start_date=start_date,
                                                                   end_date=end_date,
                                                                   datetime_dtype="datetime64[s]",
                                                                   timedelta_dtype="timedelta64[s]")

    # save to file
    data_dict = {
        "Person ID": person_ids,
        "Insertion Time": insertion_times,
        "Deletion Time": deletion_times
    }
    df = pd.DataFrame(data=data_dict)

    if path is None:
        path = f"./fake_random_dataset_{num_rows}.csv"
    df.to_csv(path, index=False)

    domain_json = json.dumps({})
    if domain_path is None:
        domain_path = f"./fake_random_dataset_{num_rows}_domain.json"
    with open(domain_path, "w") as domain_file:
        domain_file.write(domain_json)


def create_fake_ins_after_del_dataset(num_ins=2 ** 20, num_repeats=6, path=None, domain_path=None):
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
        path = f"./fake_ins_after_del_dataset_{num_ins}_{num_repeats}.csv"
    df.to_csv(path, index=False)

    domain_json = json.dumps({})
    if domain_path is None:
        domain_path = f"./fake_ins_after_del_dataset_{num_ins}_{num_repeats}_domain.json"
    with open(domain_path, "w") as domain_file:
        domain_file.write(domain_json)


def get_random_ins_and_del_times(num_rows, start_date, end_date, datetime_dtype, timedelta_dtype):
    possible_timestamps = np.arange(start_date, end_date, dtype=datetime_dtype)
    insertion_times = np.random.choice(possible_timestamps, size=num_rows)

    # generate random deletion times for each person (deletion times should be after insertion times)
    time_deltas = np.array([timedelta(days=np.random.randint(1, 30)) for _ in range(num_rows)],
                           dtype=timedelta_dtype)
    deletion_times = insertion_times + time_deltas
    end_date = datetime.date(end_date)  # both deletion_times[i] and end_date need to have the same type to be compared
    deletion_times = np.where(deletion_times > end_date, end_date, deletion_times)
    return insertion_times, deletion_times


def get_config_for_adult_dataset(domain, size, enc_type):
    if size == "small":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "age": "categorical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "categorical",
                    "income": "categorical"
                },
                "transformers": {
                    "age": None,
                    "race": OrderedLabelEncoder(order=domain["race"]),
                    "sex": OrderedLabelEncoder(order=domain["sex"]),
                    "hours-per-week": None,
                    "income": OrderedLabelEncoder(order=domain["income"])
                }
            }
        elif enc_type == "binarized":
            return {
                "sdtypes": {
                    "age": "numerical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "numerical",
                    "income": "categorical"
                },
                "transformers": {
                    "age": None,
                    "race": OrderedLabelEncoder(order=domain["race"]),
                    "sex": OrderedLabelEncoder(order=domain["sex"]),
                    "hours-per-week": None,
                    "income": OrderedLabelEncoder(order=domain["income"])
                }
            }
    elif size == "medium":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "age": "categorical",
                    "workclass": "categorical",
                    "occupation": "categorical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "categorical",
                    "income": "categorical"
                },
                "transformers": {
                    "age": None,
                    "workclass": OrderedLabelEncoder(order=domain["workclass"]),
                    "occupation": OrderedLabelEncoder(order=domain["occupation"]),
                    "race": OrderedLabelEncoder(order=domain["race"]),
                    "sex": OrderedLabelEncoder(order=domain["sex"]),
                    "hours-per-week": None,
                    "income": OrderedLabelEncoder(order=domain["income"])
                }
            }
        elif enc_type == "binarized":
            return {
                "sdtypes": {
                    "age": "numerical",
                    "workclass": "categorical",
                    "occupation": "categorical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "numerical",
                    "income": "categorical"
                },
                "transformers": {
                    "age": None,
                    "workclass": OrderedLabelEncoder(order=domain["workclass"]),
                    "occupation": OrderedLabelEncoder(order=domain["occupation"]),
                    "race": OrderedLabelEncoder(order=domain["race"]),
                    "sex": OrderedLabelEncoder(order=domain["sex"]),
                    "hours-per-week": None,
                    "income": OrderedLabelEncoder(order=domain["income"])
                }
            }


def create_adult_dataset(path, domain_path, size, enc_type):
    # read dataset and domain
    df = pd.read_csv(path, na_values='?')
    with open(domain_path, "r") as domain_file:
        domain = json.load(domain_file)

    # drop columns not in the domain
    columns_to_keep = list(domain.keys())
    df = df[columns_to_keep]

    # remove rows if any of the features are NaN / missing
    df = df.dropna(how="any")

    if enc_type == "ohe":
        # make continuous columns categorical (integers)
        df["age"] = pd.cut(df["age"], bins=domain["age"], ordered=True, labels=False)
        df["hours-per-week"] = pd.cut(df["hours-per-week"], bins=domain["hours-per-week"], ordered=True, labels=False)

    # get RDT transformer config based on 'size' and 'enc_type'
    ht_config = get_config_for_adult_dataset(domain, size, enc_type)
    ht = HyperTransformer()
    ht.set_config(config=ht_config)
    df = ht.fit_transform(df)

    # add random insertion and deletion times
    start_date = datetime(year=2023, month=1, day=1)
    end_date = datetime(year=2023, month=12, day=31)
    insertion_times, deletion_times = get_random_ins_and_del_times(num_rows=df.shape[0],
                                                                   start_date=start_date,
                                                                   end_date=end_date,
                                                                   datetime_dtype="datetime64[D]",
                                                                   timedelta_dtype="timedelta64[D]")
    df["Insertion Time"] = insertion_times
    df["Deletion Time"] = deletion_times

    # save processed dataset
    df.to_csv(f"./adult_{size}_{enc_type}.csv", index_label="Person ID")


if __name__ == "__main__":
    adult_dataset_path = f"./adult.csv"

    adult_size = "medium"
    encoding_type = "binarized"
    adult_dataset_domain_path = f"./adult_{adult_size}_{encoding_type}_domain.json"

    create_adult_dataset(path=adult_dataset_path,
                         domain_path=adult_dataset_domain_path,
                         size=adult_size,
                         enc_type=encoding_type)
