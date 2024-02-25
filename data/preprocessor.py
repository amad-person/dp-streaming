import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from rdt import HyperTransformer
from rdt.transformers import OrderedLabelEncoder
from folktables import ACSDataSource, ACSHealthInsurance, ACSPublicCoverage


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


def get_deterministic_ins_and_del_times(num_rows, start_date, batch_size, window_size):
    num_batches = num_rows // batch_size
    print(num_batches)
    insertion_times, deletion_times = [], []
    for r in range(num_batches):
        insertion_time = start_date + timedelta(days=r)
        insertion_times += [insertion_time] * batch_size
        deletion_time = insertion_time + timedelta(days=window_size)
        deletion_times += [deletion_time] * batch_size

    # add any remaining rows to the last batch
    num_remaining_rows = num_rows - (num_batches * batch_size)
    insertion_time = start_date + timedelta(days=num_batches - 1)
    insertion_times += [insertion_time] * num_remaining_rows
    deletion_time = insertion_time + timedelta(days=window_size)
    deletion_times += [deletion_time] * num_remaining_rows

    assert num_rows == len(insertion_times) and num_rows == len(deletion_times)
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
                    # "age": "numerical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "numerical",
                    "income": "categorical"
                },
                "transformers": {
                    # "age": None,
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
    elif size == "full":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "age": "categorical",
                    "workclass": "categorical",
                    "education": "categorical",
                    "educational-num": "categorical",
                    "occupation": "categorical",
                    "relationship": "categorical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "categorical",
                    "income": "categorical"
                },
                "transformers": {
                    "age": None,
                    "workclass": OrderedLabelEncoder(order=domain["workclass"]),
                    "education": OrderedLabelEncoder(order=domain["education"]),
                    "educational-num": OrderedLabelEncoder(order=domain["educational-num"]),
                    "occupation": OrderedLabelEncoder(order=domain["occupation"]),
                    "relationship": OrderedLabelEncoder(order=domain["relationship"]),
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
                    "education": "categorical",
                    "educational-num": "categorical",
                    "occupation": "categorical",
                    "relationship": "categorical",
                    "race": "categorical",
                    "sex": "categorical",
                    "hours-per-week": "numerical",
                    "income": "categorical"
                },
                "transformers": {
                    "age": None,
                    "workclass": OrderedLabelEncoder(order=domain["workclass"]),
                    "education": OrderedLabelEncoder(order=domain["education"]),
                    "educational-num": OrderedLabelEncoder(order=domain["educational-num"]),
                    "occupation": OrderedLabelEncoder(order=domain["occupation"]),
                    "relationship": OrderedLabelEncoder(order=domain["relationship"]),
                    "race": OrderedLabelEncoder(order=domain["race"]),
                    "sex": OrderedLabelEncoder(order=domain["sex"]),
                    "hours-per-week": None,
                    "income": OrderedLabelEncoder(order=domain["income"])
                }
            }


def create_adult_dataset(path, domain_path, size, enc_type, batch_size, window_size):
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
        if "age" in df.columns:
            df["age"] = pd.cut(df["age"],
                               bins=domain["age"],
                               ordered=True,
                               labels=False)

        if "hours-per-week" in df.columns:
            df["hours-per-week"] = pd.cut(df["hours-per-week"],
                                          bins=domain["hours-per-week"],
                                          ordered=True,
                                          labels=False)
    elif enc_type == "binarized":
        # make continuous columns integers
        if "AGEP" in df.columns:
            df["AGEP"] = df["AGEP"].astype(int)

    # get RDT transformer config based on 'size' and 'enc_type'
    ht_config = get_config_for_adult_dataset(domain, size, enc_type)
    ht = HyperTransformer()
    ht.set_config(config=ht_config)
    df = ht.fit_transform(df)

    # add deterministic insertion and deletion times
    start_date = datetime(year=2023, month=1, day=1)
    insertion_times, deletion_times = get_deterministic_ins_and_del_times(num_rows=df.shape[0],
                                                                          start_date=start_date,
                                                                          batch_size=batch_size,
                                                                          window_size=window_size)
    df["Insertion Time"] = insertion_times
    df["Deletion Time"] = deletion_times

    # save processed dataset
    df.to_csv(f"./adult_{size}_batch{batch_size}_window{window_size}_{enc_type}.csv",
              index_label="Person ID")


def get_config_for_acs_data_health_ins_dataset(domain, size, enc_type):
    if size == "small":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "AGEP": "categorical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "HINS2": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "HINS2": OrderedLabelEncoder(order=domain["HINS2"]),
                }
            }
        elif enc_type == "binarized":
            return {
                "sdtypes": {
                    "AGEP": "numerical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "HINS2": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "HINS2": OrderedLabelEncoder(order=domain["HINS2"]),
                }
            }
    elif size == "medium":
        if enc_type == "ohe":
            pass
        elif enc_type == "binarized":
            pass
    elif size == "full":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "AGEP": "numerical",
                    "SCHL": "categorical",
                    "MAR": "categorical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "ESP": "categorical",
                    "CIT": "categorical",
                    "MIG": "categorical",
                    "MIL": "categorical",
                    "ANC": "categorical",
                    "NATIVITY": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "RACAIAN": "categorical",
                    "RACASN": "categorical",
                    "RACBLK": "categorical",
                    "RACNH": "categorical",
                    "RACPI": "categorical",
                    "RACSOR": "categorical",
                    "RACWHT": "categorical",
                    "PINCP": "numerical",
                    "ESR": "categorical",
                    "FER": "categorical",
                    "HINS2": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "SCHL": OrderedLabelEncoder(order=domain["SCHL"]),
                    "MAR" : OrderedLabelEncoder(order=domain["MAR"]),
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "ESP": OrderedLabelEncoder(order=domain["ESP"]),
                    "CIT": OrderedLabelEncoder(order=domain["CIT"]),
                    "MIG": OrderedLabelEncoder(order=domain["MIG"]),
                    "MIL": OrderedLabelEncoder(order=domain["MIL"]),
                    "ANC": OrderedLabelEncoder(order=domain["ANC"]),
                    "NATIVITY": OrderedLabelEncoder(order=domain["NATIVITY"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "RACAIAN": OrderedLabelEncoder(order=domain["RACAIAN"]),
                    "RACASN": OrderedLabelEncoder(order=domain["RACASN"]),
                    "RACBLK": OrderedLabelEncoder(order=domain["RACBLK"]),
                    "RACNH": OrderedLabelEncoder(order=domain["RACNH"]),
                    "RACPI": OrderedLabelEncoder(order=domain["RACPI"]),
                    "RACSOR": OrderedLabelEncoder(order=domain["RACSOR"]),
                    "RACWHT": OrderedLabelEncoder(order=domain["RACWHT"]),
                    "PINCP": None,
                    "ESR": OrderedLabelEncoder(order=domain["ESR"]),
                    "FER": OrderedLabelEncoder(order=domain["FER"]),
                    "HINS2": OrderedLabelEncoder(order=domain["HINS2"]),
                }
            }
        elif enc_type == "binarized":
            pass


def create_acs_health_ins_dataset(domain_path, size, acs_data, enc_type, batch_size, window_size):
    # read dataset and domain
    df, df_labels, _ = ACSHealthInsurance.df_to_pandas(acs_data)
    df["HINS2"] = df_labels["HINS2"].astype(int)
    with open(domain_path, "r") as domain_file:
        domain = json.load(domain_file)

    # drop columns not in the domain
    columns_to_keep = list(domain.keys())
    df = df[columns_to_keep]

    # remove rows if any of the features are NaN / missing
    df = df.dropna(how="any")

    if enc_type == "ohe":
        # make continuous columns categorical (integers)
        if "AGEP" in df.columns:
            df["AGEP"] = pd.cut(df["AGEP"],
                                bins=domain["AGEP"],
                                ordered=True,
                                labels=False)

        if "PINCP" in df.columns:
            df["PINCP"] = pd.cut(df["PINCP"],
                                 bins=domain["PINCP"],
                                 ordered=True,
                                 labels=False)
    elif enc_type == "binarized":
        # make continuous columns integers
        if "AGEP" in df.columns:
            df["AGEP"] = df["AGEP"].astype(int)

        if "PINCP" in df.columns:
            df["PINCP"] = df["PINCP"].astype(int)

    # get RDT transformer config based on 'size' and 'enc_type'
    ht_config = get_config_for_acs_data_health_ins_dataset(domain, size, enc_type)
    ht = HyperTransformer()
    ht.set_config(config=ht_config)
    df = ht.fit_transform(df)

    # add deterministic insertion and deletion times
    start_date = datetime(year=2023, month=1, day=1)
    insertion_times, deletion_times = get_deterministic_ins_and_del_times(num_rows=df.shape[0],
                                                                          start_date=start_date,
                                                                          batch_size=batch_size,
                                                                          window_size=window_size)

    df["Insertion Time"] = insertion_times
    df["Deletion Time"] = deletion_times

    # save processed dataset
    df.to_csv(f"./acs_health_ins_{size}_batch{batch_size}_window{window_size}_{enc_type}.csv",
              index_label="Person ID")


def get_config_for_acs_data_public_cov_dataset(domain, size, enc_type):
    if size == "small":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "AGEP": "numerical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "PUBCOV": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "PUBCOV": OrderedLabelEncoder(order=domain["PUBCOV"]),
                }
            }
        elif enc_type == "binarized":
            return {
                "sdtypes": {
                    "AGEP": "numerical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "PUBCOV": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "PUBCOV": OrderedLabelEncoder(order=domain["PUBCOV"]),
                }
            }
    elif size == "medium":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "AGEP": "numerical",
                    "MAR": "categorical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "PINCP": "numerical",
                    "ESR": "categorical",
                    "FER": "categorical",
                    "PUBCOV": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "MAR": OrderedLabelEncoder(order=domain["MAR"]),
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "PINCP": None,
                    "ESR": OrderedLabelEncoder(order=domain["ESR"]),
                    "FER": OrderedLabelEncoder(order=domain["FER"]),
                    "PUBCOV": OrderedLabelEncoder(order=domain["PUBCOV"]),
                }
            }
        elif enc_type == "binarized":
            pass
    elif size == "full":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "AGEP": "categorical",
                    "SCHL": "categorical",
                    "MAR": "categorical",
                    "SEX": "categorical",
                    "DIS": "categorical",
                    "ESP": "categorical",
                    "CIT": "categorical",
                    "MIG": "categorical",
                    "MIL": "categorical",
                    "ANC": "categorical",
                    "NATIVITY": "categorical",
                    "DEAR": "categorical",
                    "DEYE": "categorical",
                    "DREM": "categorical",
                    "PINCP": "categorical",
                    "ESR": "categorical",
                    "FER": "categorical",
                    "PUBCOV": "categorical"
                },
                "transformers": {
                    "AGEP": None,
                    "SCHL": OrderedLabelEncoder(order=domain["SCHL"]),
                    "MAR" : OrderedLabelEncoder(order=domain["MAR"]),
                    "SEX": OrderedLabelEncoder(order=domain["SEX"]),
                    "DIS": OrderedLabelEncoder(order=domain["DIS"]),
                    "ESP": OrderedLabelEncoder(order=domain["ESP"]),
                    "CIT": OrderedLabelEncoder(order=domain["CIT"]),
                    "MIG": OrderedLabelEncoder(order=domain["MIG"]),
                    "MIL": OrderedLabelEncoder(order=domain["MIL"]),
                    "ANC": OrderedLabelEncoder(order=domain["ANC"]),
                    "NATIVITY": OrderedLabelEncoder(order=domain["NATIVITY"]),
                    "DEAR": OrderedLabelEncoder(order=domain["DEAR"]),
                    "DEYE": OrderedLabelEncoder(order=domain["DEYE"]),
                    "DREM": OrderedLabelEncoder(order=domain["DREM"]),
                    "PINCP": None,
                    "ESR": OrderedLabelEncoder(order=domain["ESR"]),
                    "FER": OrderedLabelEncoder(order=domain["FER"]),
                    "PUBCOV": OrderedLabelEncoder(order=domain["PUBCOV"]),
                }
            }
        elif enc_type == "binarized":
            pass


def create_acs_public_cov_dataset(domain_path, size, acs_data, enc_type, batch_size, window_size):
    # read dataset and domain
    df, df_labels, _ = ACSPublicCoverage.df_to_pandas(acs_data)
    df["PUBCOV"] = df_labels["PUBCOV"].astype(int)
    with open(domain_path, "r") as domain_file:
        domain = json.load(domain_file)

    # drop columns not in the domain
    columns_to_keep = list(domain.keys())
    df = df[columns_to_keep]

    # remove rows if any of the features are NaN / missing
    df = df.dropna(how="any")

    if enc_type == "ohe":
        # make continuous columns categorical (integers)
        if "AGEP" in df.columns:
            df["AGEP"] = pd.cut(df["AGEP"],
                                bins=domain["AGEP"],
                                ordered=True,
                                labels=False)

        if "PINCP" in df.columns:
            df["PINCP"] = pd.cut(df["PINCP"],
                                 bins=domain["PINCP"],
                                 ordered=True,
                                 labels=False)
    elif enc_type == "binarized":
        # make continuous columns integers
        if "AGEP" in df.columns:
            df["AGEP"] = df["AGEP"].astype(int)

        if "PINCP" in df.columns:
            df["PINCP"] = df["PINCP"].astype(int)

    # get RDT transformer config based on 'size' and 'enc_type'
    ht_config = get_config_for_acs_data_public_cov_dataset(domain, size, enc_type)
    ht = HyperTransformer()
    ht.set_config(config=ht_config)
    df = ht.fit_transform(df)

    # add deterministic insertion and deletion times
    start_date = datetime(year=2023, month=1, day=1)
    insertion_times, deletion_times = get_deterministic_ins_and_del_times(num_rows=df.shape[0],
                                                                          start_date=start_date,
                                                                          batch_size=batch_size,
                                                                          window_size=window_size)

    df["Insertion Time"] = insertion_times
    df["Deletion Time"] = deletion_times

    # save processed dataset
    df.to_csv(f"./acs_public_cov_{size}_batch{batch_size}_window{window_size}_{enc_type}.csv",
              index_label="Person ID")


def get_config_for_ny_taxi_dataset(domain, size, enc_type):
    if size == "medium":
        if enc_type == "ohe":
            return {
                "sdtypes": {
                    "VendorID": "categorical",
                    "passenger_count": "categorical",
                    "RatecodeID": "categorical",
                    "store_and_fwd_flag": "categorical",
                    "payment_type": "categorical",
                },
                "transformers": {
                    "VendorID": OrderedLabelEncoder(order=domain["VendorID"]),
                    "passenger_count": None,
                    "RatecodeID": OrderedLabelEncoder(order=domain["RatecodeID"]),
                    "store_and_fwd_flag": OrderedLabelEncoder(order=domain["store_and_fwd_flag"]),
                    "payment_type": OrderedLabelEncoder(order=domain["payment_type"]),
                }
            }
        elif enc_type == "binarized":
            return {
                "sdtypes": {
                    "VendorID": "categorical",
                    "passenger_count": "numerical",
                    "RatecodeID": "categorical",
                    "store_and_fwd_flag": "categorical",
                    "payment_type": "categorical",
                },
                "transformers": {
                    "VendorID": OrderedLabelEncoder(order=domain["VendorID"]),
                    "passenger_count": None,
                    "RatecodeID": OrderedLabelEncoder(order=domain["RatecodeID"]),
                    "store_and_fwd_flag": OrderedLabelEncoder(order=domain["store_and_fwd_flag"]),
                    "payment_type": OrderedLabelEncoder(order=domain["payment_type"]),
                }
            }


def create_ny_taxi_dataset(path, year, domain_path, size, enc_type, batch_size=None, window_size=None):
    # read dataset and domain
    df = pd.read_parquet(path)
    with open(domain_path, "r") as domain_file:
        domain = json.load(domain_file)

    # select rows with pick up times in specified year
    df["tpep_pickup_datetime"] = df["tpep_pickup_datetime"].astype("datetime64[ns]")
    df = df[df["tpep_pickup_datetime"].dt.year == year]

    # sort by pick up time
    df = df.sort_values(by="tpep_pickup_datetime", ascending=True).reset_index()

    # keep timestamp columns
    insertion_times = df["tpep_pickup_datetime"]
    deletion_times = df["tpep_dropoff_datetime"]

    # drop columns not in the domain
    columns_to_keep = list(domain.keys())
    df = df[columns_to_keep]

    # remove rows if any of the features are NaN / missing
    df = df.dropna(how="any")

    if enc_type == "ohe":
        # make continuous columns categorical (integers)
        if "passenger_count" in df.columns:
            df["passenger_count"] = pd.cut(df["passenger_count"],
                                           bins=domain["passenger_count"],
                                           ordered=True,
                                           labels=False)
    elif enc_type == "binarized":
        # make continuous columns integers
        if "passenger_count" in df.columns:
            df["passenger_count"] = df["passenger_count"].astype(int)

    # get RDT transformer config based on 'size' and 'enc_type'
    ht_config = get_config_for_ny_taxi_dataset(domain, size, enc_type)
    ht = HyperTransformer()
    ht.set_config(config=ht_config)
    df = ht.fit_transform(df)

    # process pick up and drop off times
    df["Insertion Time"] = insertion_times
    df["Deletion Time"] = deletion_times

    # save processed dataset
    df.to_csv(f"./ny_taxi_{size}_batch{batch_size}_window{window_size}_{enc_type}.csv",
              index_label="Person ID")


if __name__ == "__main__":
    ny_taxi_data_year = 2023
    ny_taxi_data_path = f"./yellow_tripdata_2023-01.parquet"
    ny_taxi_data_size = "medium"
    encoding_type = "binarized"
    ny_taxi_data_domain_path = f"./ny_taxi_{ny_taxi_data_size}_{encoding_type}_domain.json"
    create_ny_taxi_dataset(path=ny_taxi_data_path,
                           year=ny_taxi_data_year,
                           domain_path=ny_taxi_data_domain_path,
                           size=ny_taxi_data_size,
                           enc_type=encoding_type)

    # data_source = ACSDataSource(survey_year='2018',
    #                             horizon='1-Year',
    #                             survey='person')
    # acs_data = data_source.get_data(states=["NY"], download=True)
    # acs_data_subset = "public_cov"
    # acs_data_size = "medium"
    # encoding_type = "ohe"
    # for batch_size in [5, 10, 25, 50]:
    #     for window_size in [1, 3, 5, 10]:
    #         acs_data_domain_path = f"./acs_{acs_data_subset}_{acs_data_size}_{encoding_type}_domain.json"
    #         create_acs_public_cov_dataset(domain_path=acs_data_domain_path,
    #                                       size=acs_data_size,
    #                                       acs_data=acs_data,
    #                                       enc_type=encoding_type,
    #                                       batch_size=batch_size,
    #                                       window_size=window_size)

    # adult_dataset_path = f"./adult.csv"
    # adult_size = "small"
    # encoding_type = "ohe"
    # adult_dataset_domain_path = f"./adult_{adult_size}_{encoding_type}_domain.json"
    # for batch_size in [5]:
    #     for window_size in [1, 3, 5, 10]:
    #         create_adult_dataset(path=adult_dataset_path,
    #                              domain_path=adult_dataset_domain_path,
    #                              size=adult_size,
    #                              enc_type=encoding_type,
    #                              batch_size=batch_size,
    #                              window_size=window_size)
