from warnings import warn

from imblearn.over_sampling.smote import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


def _per_patient_smote(x, y):
    x_rows = []
    y_obs = []
    x.patient.str.replace('-\d', '')
    for patient in x.patient.unique():
        pt_idx = x[x.patient == patient].index
        x_slice = x.loc[pt_idx]
        x_slice = x_slice.drop('patient', 1)
        y_slice = y.loc[pt_idx]
        dbl_samples = len(y_slice[y_slice == 1])
        bs_samples = len(y_slice[y_slice == 2])
        if dbl_samples < 2:
            x_rows.extend(x_slice.values)
            y_obs.extend(y_slice.values)
            continue
        elif bs_samples < 2:
            x_rows.extend(x_slice.values)
            y_obs.extend(y_slice.values)
            continue
        else:
            smote = SMOTE(k=5 if dbl_samples >= 5 else dbl_samples-1, ratio=1.0)
        cols = x_slice.columns
        new_x, new_y = smote.fit_sample(x_slice, y_slice)

        for _ in range(len(y_slice.unique()[2:])):
            new_x, new_y = smote.fit_sample(new_x, new_y)
        x_rows.extend(new_x)
        y_obs.extend(new_y)
    new_x, new_y = DataFrame(x_rows, columns=cols), Series(y_obs)
    new_x = new_x.reindex(np.random.permutation(new_x.index))
    new_y = new_y.loc[new_x.index]
    return new_x, new_y


def _apply_smote(x, y, ratio):
    for label in y.unique():
        if len(y[y == label]) < 2:
            raise Exception("You do not have sufficient asynchrony observations for label: {}!".format(label))
    # 50/50 split at first.
    n_samples = len(y[y == 1])
    class_counts = y.value_counts()
    majority_samples = class_counts.max()
    samples = {class_counts.argmax(): class_counts.max()}
    for cls, count in dict(class_counts).items()[1:]:
        if int(ratio * majority_samples) < class_counts[cls]:
            samples[cls] = class_counts[cls]
        else:
            samples[cls] = int(ratio * majority_samples)
    if n_samples <= 5:
        k = len(y[y == 1])
        smote = SMOTE(k=k - 1, ratio=ratio)
    else:
        smote = SMOTE(ratio=samples)
    try:
        x = x.drop('patient', 1)
    except:
        pass
    print("Original samples before SMOTE: {}".format(n_samples))
    # for some reason I set k=n_samples but it thinks its n_samples + 1
    # so I need to adjust down 1
    cols = x.columns
    new_x, new_y = smote.fit_sample(x, y)
    for _ in range(len(y.unique()[2:])):
        new_x, new_y = smote.fit_sample(new_x, new_y)
    new_x = DataFrame(new_x)
    new_y = Series(new_y)
    # reindex because the results will look like [0, 0, 0, ..., 1, 1, 1, ...]
    new_x = new_x.reindex(np.random.permutation(new_x.index))
    new_y = new_y.loc[new_x.index]
    new_x.columns = cols
    return new_x, new_y


def _apply_under_sampling(df, y, ratio):
    try:
        df = df.drop('patient', 1)
    except:
        pass
    cols = df.columns
    rus = RandomUnderSampler(ratio=ratio, return_indices=True)
    x_resampled, y_resampled, idx_resampled = rus.fit_sample(df, y)
    x = DataFrame(x_resampled, columns=cols)
    y = Series(y_resampled)
    return x, y


def _fa_obs_splitter(df, y, n_samples, with_smote, smote_ratio):
    if with_smote:
        df, y = _apply_smote(df, y, smote_ratio)

    async_index = y[y == 1].index[0:n_samples]
    if len(async_index) == 0:
        raise Exception("No asynchrony samples exist!")
    train_index = []
    for i in df.index:
        train_index.append(i)
        if i == async_index[-1]:
            break
    if len(y[y == 1]) <= n_samples:
        raise Exception(
            "The data set does not have enough asynchrony examples! Current #: "
            "{}, # necessary : {}".format(len(y[y == 1]), n_samples + 1)
        )
    test_index = df.index.difference(train_index)
    return [(df.loc[train_index], df.loc[test_index], y.loc[train_index], y.loc[test_index])]


def _fa_obs_per_patient_splitter(df, y, patient_to_use, n_samples, with_smote, smote_ratio):
    index = [i for i in df.index if patient_to_use in i]
    df, y = df.loc[index], y.loc[index]
    return _fa_obs_splitter(df, y, n_samples, with_smote, smote_ratio)


def select_n_async_per_patient_split(df, y, args):
    return _fa_obs_per_patient_splitter(df, y, args.patient_to_use, args.n_samples, args.with_smote, args.smote_ratio)


def select_n_async_cross_patient_split(df, y, args):
    return _fa_obs_splitter(df, y, args.n_samples, args.with_smote, args.smote_ratio)


def simple_split(df, y, args):
    with_smote = args.with_smote
    split_ratio = args.split_ratio

    if with_smote:
        df, y = _apply_smote(df, y, args.smote_ratio)
    return [(train_test_split(df, y, test_size=split_ratio))]


def per_patient_smote_on_train_split(df, y, args):
    patient_to_use = args.patient_to_use
    n_samples = args.n_samples

    x_train, x_test, y_train, y_test = _fa_obs_per_patient_splitter(df, y, patient_to_use, n_samples, False, None)[0]
    x_train, y_train = _apply_smote(x_train, y_train, args.smote_ratio)
    return [(x_train, x_test, y_train, y_test)]


def per_patient_split(df, y, args):
    index = [i for i in df.index if args.patient_to_use in i]
    df, y = df.loc[index], y.loc[index]
    if args.with_smote:
        df, y = _apply_smote(df, y, args.smote_ratio)
    return [(train_test_split(df, y, test_size=args.split_ratio))]


def per_patient_time_interval(df, y, args):
    patient_to_use = args.patient_to_use
    train_time = args.train_time

    def time_interval_generator(df, y, index):
        lower_bound = 0
        upper_bound = train_time * 60
        interval = upper_bound
        while lower_bound < float(index[-1].split("-")[-1]):
            train_index = [i for i in index if lower_bound < float(i.split("-")[-1]) < upper_bound]
            test_index = index - train_index
            yield [(df.loc[train_index], y.loc[train_index], df.loc[test_index], y.loc[test_index])]
            lower_bound = upper_bound
            upper_bound = upper_bound + interval

    index = [i for i in df.index if patient_to_use in i]
    df, y = df.loc[index], y.loc[index]
    return time_interval_generator(df, y, df.index)


def cross_patient_kfold_split(df, y, args):
    n_folds = args.folds
    with_smote = args.with_smote

    def kfold_generator(patients):
        patients_per_fold = float(len(patients)) / n_folds
        for fold_idx in range(n_folds):
            lower_bound = int(round(fold_idx * patients_per_fold))
            upper_bound = int(round((fold_idx + 1) * patients_per_fold))
            to_use = list(patients)[lower_bound:upper_bound]
            test_index = [i for i in df.index if i.split("-")[1] in to_use]
            x_test, y_test = df.loc[test_index], y.loc[test_index]
            x_train, y_train  = df.loc[df.index.difference(test_index)], y.loc[df.index.difference(test_index)]
            if with_smote:
                try:
                    x_train, y_train = _apply_smote(x_train, y_train, args.smote_ratio)
                except Exception as err:
                    warn("Skipping patient kfold due to {}".format(err))
                    continue

            if args.undersample:
                x_train, y_train = _apply_under_sampling(x_train, y_train, args.undersample)

            if len(y_test[y_test != 0]) < 1:
                warn("Skipping patient {} kfold due to no test samples".format(to_use))
                continue
            print("Using test patients: {}".format(", ".join(to_use)))
            yield (x_train, x_test, y_train, y_test)

    patients = set(map(lambda x: x.split("-")[1], df.index))
    if n_folds > len(patients):
        raise Exception("Cannot have more folds than patients!")
    return kfold_generator(patients)


def idx_cross_patient_kfold_split(df, y, n_folds):
    def kfold_generator(patients):
        patients_per_fold = float(len(patients)) / n_folds
        tmp_df = df.copy()
        tmp_df.index = range(len(tmp_df))
        for i in range(n_folds):
            lower_bound = int(round(i * patients_per_fold))
            upper_bound = int(round((i + 1) * patients_per_fold))
            to_use = list(patients)[lower_bound:upper_bound]

            test_index = [idx for idx, i in enumerate(df.index) if i.split("-")[1] in to_use]
            train_index = tmp_df.index.difference(test_index)
            yield (train_index, test_index)

    patients = set(map(lambda x: x.split("-")[1], df.index))
    if n_folds > len(patients):
        raise Exception("Cannot have more folds than patients!")
    return kfold_generator(patients)


def cross_patient_n_samples_split(df, y, args):
    """
    Function takes a patient to test on and then trains on the remaining patients
    in the set. We sample a certain number of FA observations from the test patient
    and then train on the remainder of the test set.
    """
    train_patients = set(map(lambda x: x.split("-")[1], df.index)).difference([args.patient_to_use])
    train_obs = df.query("patient in {}".format(list(train_patients)))
    test_patient_obs = df[df['patient'] == args.patient_to_use]
    add_x_train, x_test, add_y_train, y_test = _fa_obs_per_patient_splitter(
        test_patient_obs, y.loc[test_patient_obs.index], args.patient_to_use, args.n_samples, False, None
    )[0]
    x_train = train_obs.append(add_x_train)
    y_train = y.loc[x_train.index]
    if args.with_smote:
        x_train, y_train = _apply_smote(x_train, y_train, args.smote_ratio)
    return [(x_train, x_test, y_train, y_test)]


def clustering_split(df, y, args):
    if args.with_smote:
        df, y = _apply_smote(df, y, args.smote_ratio)
    return [([], df, [], y)]


def train_on_everything_split(df, y, args):
    if args.with_smote:
        df, y = _apply_smote(df, y, args.smote_ratio)
    return [(df, [], y, [])]


def per_patient_clustering_split(df, y, args):
    index = [i for i in df.index if args.patient_to_use in i]
    df, y = df.loc[index], y.loc[index]
    if args.with_smote:
        df, y = _apply_smote(df, y, args.smote_ratio)
    return [([], df, [], y)]
