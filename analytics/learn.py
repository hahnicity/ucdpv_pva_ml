from argparse import ArgumentParser
import csv
from operator import xor
import os
import pickle
from warnings import warn

import numpy as np
from pandas import DataFrame, read_pickle
from scipy.stats.mstats import winsorize
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from analyze import run_and_print_chi2, run_chi2
from classifier import Classifier, CLASSIFIER_DICT
from collate import (
    get_all_greg_selected_features,
    get_binary_target_basic_dbl_trigger_data,
    get_binary_target_bs_patient_data,
    get_binary_target_cough_data,
    get_binary_target_deriv_valid_bs_patient_data,
    get_binary_target_deriv_valid_dbl_trigger_data,
    get_binary_target_fa_patient_data,
    get_binary_target_suction_data,
    get_binary_target_retrospective_dbl_trigger_data,
    get_bs_all,
    get_bs_chi2,
    get_bs_curated,
    get_bs_heuristic,
    get_bs_dbl_non_retrospective_data,
    get_bs_dbl_retrospective_data,
    get_bs_dbl_retrospective_data_first_dbl_as_bs,
    get_bs_dbl_retrospective_first_dbl_as_bs_deriv_valid,
    get_bs_dbl_retrospective_first_dbl_as_norm_deriv_valid,
    get_bs_dbl_retrospective_data_first_dbl_as_bs_val_cohort,
    get_chi2_optimum,
    get_co_all,
    get_co_curated_no_tvi,
    get_co_curated_with_tvi,
    get_co_heuristic,
    get_dbl_curated,
    get_dbl_chi2,
    get_dbl_retro_chi2,
    get_dbl_trigger_heuristic_features,
    get_dbl_trigger_heuristic_and_metadata_features,
    get_dbl_trigger_retrospective_plus_metadata_features,
    get_derived_metadata_features,
    get_fa_heuristic,
    get_integrated_pressure_curve_df,
    get_multi_class_binary_label_retrospective_data,
    get_multi_retrospective_data_fdb_deriv_val_cohort,
    get_multi_target_derivation_cohort_data,
    get_multi_target_retrospective_data,
    get_multi_target_retrospective_data_first_dbl_as_bs,
    get_retro_prev_plus_metadata,
    get_retro_prev_prev_plus_metadata,
    get_retro_plus_metadata,
    get_retro_non_noisy,
    get_retro_stripped_expert_plus_chi2,
    get_retro_stripped_expert_plus_chi2_2,
    get_retro_stripped_expert_plus_chi2_3,
    get_retro_stripped_expert_plus_chi2_4,
    get_retro_stripped_expert_plus_chi2_5,
    get_retro_stripped_expert_plus_chi2_6,
    get_retro_stripped_expert_plus_chi2_7,
    get_retro_stripped_expert_plus_instrr,
    get_retro_stripped_expert_plus_instrr_prev,
    get_retro_stripped_lower_prec,
    get_retro_stripped_higher_prec,
    get_settings_and_derived_features,
    get_slopes_of_pressure_curve_df,
    get_suction_all,
    get_suction_curated,
    get_suction_heuristic,
    get_v1,
    get_v2,
    get_v2_and_metadata,
    get_vent_settings_features,
)
from train_test_set import (
    clustering_split,
    cross_patient_kfold_split,
    cross_patient_n_samples_split,
    per_patient_clustering_split,
    per_patient_split,
    per_patient_time_interval,
    select_n_async_cross_patient_split,
    select_n_async_per_patient_split,
    per_patient_smote_on_train_split,
    simple_split,
    train_on_everything_split,
)

class NoopScaler(object):
    def fit(self, x):
        return x

    def transform(self, x):
        return x

    def fit_transform(self, x):
        return x


SCALER_DICT = {
    "min_max": MinMaxScaler,
    "robust": RobustScaler,
    "noop": NoopScaler,
}


def get_x_y(feature_func, bins, pickle_file, new_pickling_file, gold_stnd_func, new_csv_file):
    if pickle_file and not new_pickling_file:
        df = read_pickle(pickle_file)
    elif pickle_file and new_pickling_file:
        raise Exception(
            "You cannot create a new pickle file and extract from one "
            "at the same time!"
        )
    else:
        df = feature_func(bins, gold_stnd_func)
    if new_pickling_file:
        new_pickling_file = "df/{}.pickle".format(new_pickling_file.replace(".pickle", ""))
        df.to_pickle(new_pickling_file)
    if new_csv_file:
        new_csv_file = "{}.csv".format(new_csv_file)
        df.to_csv(new_csv_file)
    # Just rename bn columns to vent_bn. Don't bother redoing all the files.
    if "bn" in df.columns:
        df = df.rename(columns={"bn": "vent_bn"})
        df.to_pickle(pickle_file)
    y = df['y']
    del df['y']
    extra_cols = ["vent_bn", "filename"]
    try:
        extra_info = df[extra_cols]
        df = df.drop(extra_cols, 1)
    except:
        extra_info = DataFrame([])
    try:
        df = df.drop(['rel_bn'], 1)
    except:
        pass
    if len(y[y != 0]) < 2:
        raise Exception(
            "You need at least 2 asynchrony observations to actually learn anything!"
        )
    return df, y, extra_info


def perform_chi2_feature_pruning(x_train, x_test, y_train, n_features, write_chi2_results, filename, patient):
    results = run_and_print_chi2(x_train, y_train)
    in_order = sorted(results, key=lambda x: x[1])[::-1]
    to_use = [row for row in in_order if not np.isnan(row[1])][:n_features]
    cols = map(lambda x: x[0], to_use)
    if write_chi2_results:
        if not patient:
            patient = "cross-patient"
        # ok so its not completely correct but whatever
        outfile = "results/chi2-feature-picker-{}-{}-{}.csv".format(n_features, os.path.basename(filename), patient)
        f = open(outfile, "w")
        writer = csv.writer(f)
        writer.writerow(["feature", "x2", "p-val", "cramers"])
        for row in results:
            if row[0] in cols:
                writer.writerow(row)
        f.flush()
        f.close()
    print("Chi2 optimum features: {}".format(cols))
    return x_train[cols], x_test[cols]


def write_results(results, cli_args):
    if cli_args.chi2_pruning:
        # XXX Janky for now. I should consider re-doing this as a split type
        # or add additional arguments for write-results
        bin_type, identifier = cli_args.chi2_pruning, cli_args.patient
    elif cli_args.split_func == cross_patient_n_samples_split:
        bin_type, identifier = cli_args.n_samples, cli_args.patient
    elif cli_args.split_func == per_patient_smote_on_train_split:
        bin_type, identifier = cli_args.n_samples, cli_args.patient
    elif cli_args.split_func == per_patient_time_interval:
        bin_type, identifier = cli_args.train_time, cli_args.patient
    elif cli_args.split_func == cross_patient_kfold_split:
        bin_type, identifier = "no_bin", "all"
    elif cli_args.split_func == select_n_async_cross_patient_split:
        bin_type, identifier = cli_args.n_samples, "all"
    elif cli_args.split_func == per_patient_split:
        bin_type, identifier = cli_args.bins, cli_args.patient
    elif cli_args.split_func == select_n_async_per_patient_split:
        bin_type, identifier = cli_args.n_samples, cli_args.patient
    elif cli_args.split_func ==  per_patient_clustering_split:
        bin_type, identifier = cli_args.n_clust, cli_args.patient
    else:
        raise Exception("{} not supported for results writing!".format(
            cli_args.split_func.__name__
        ))
    if cli_args.is_binning_analysis:
        bin_type = cli_args.bins

    if not cli_args.outfile:
        outfile = "results-{}-{}-{}-{}.csv".format(
            cli_args.split_func.__name__, cli_args.bins, bin_type, identifier
        )
    else:
        outfile = cli_args.outfile
    with open("results/{}".format(outfile), "w") as out_:
        writer = csv.writer(out_)
        for fold in results:
            for row in fold:
                writer.writerows(row)


def get_fa_elements(df, typeof):
    if len(df) == 0:
        return 0, 0
    ctrl = len(df[df == 0])
    print "{} set has control elements: {}".format(typeof, ctrl)
    for cls in set(sorted(df.unique())).difference(set([0])):
        async_ = len(df[df == cls])
        print "{} set has # of class {} elements: {}".format(typeof, cls, async_)
    return ctrl, async_


def set_subparser_defaults(subparser, split_func):
    subparser.set_defaults(
        split_func=split_func,
        folds=None,
        patient=None,
        train_time=None,
        n_samples=None,
        split_ratio=None,
        with_smote=None,
        smote_ratio=1.0,
        undersample=None,
        n_features=None,
        train_on_all=None,
        only_patient=None,
    )


def build_parser(async_dict, feature_dict):
    parser = ArgumentParser()

    #subparsers
    subparsers = parser.add_subparsers()
    split_subparser = subparsers.add_parser("simple")
    set_subparser_defaults(split_subparser, simple_split)
    split_subparser.add_argument("--split-ratio", type=float, default=0.2)
    split_subparser.add_argument("--with-smote", action='store_true')

    clustering_subparser = subparsers.add_parser("clustering")
    set_subparser_defaults(clustering_subparser, clustering_split)
    clustering_subparser.add_argument("--with-smote", action="store_true")

    patient_clustering_subparser = subparsers.add_parser("per_patient_clustering")
    set_subparser_defaults(patient_clustering_subparser, per_patient_clustering_split)
    patient_clustering_subparser.add_argument("patient")
    patient_clustering_subparser.add_argument("--with-smote", action="store_true")

    split_with_smote_subparser = subparsers.add_parser("per_patient_smote_on_train")
    set_subparser_defaults(split_with_smote_subparser, per_patient_smote_on_train_split)
    split_with_smote_subparser.add_argument("--n-samples", required=True, type=int)
    split_with_smote_subparser.add_argument("patient")

    cross_patient = subparsers.add_parser("cross_patient_kfold")
    set_subparser_defaults(cross_patient, cross_patient_kfold_split)
    cross_patient.add_argument("--folds", type=int, default=3)
    cross_patient.add_argument('--smote-ratio', default=1.0, type=float)
    cp_mutex = cross_patient.add_mutually_exclusive_group()
    cp_mutex.add_argument("--with-smote", action='store_true')
    cross_patient.add_argument("--only-patient", help="only run a single patient's kfold. Useful for debugging")
    cp_mutex.add_argument('--undersample', choices=['auto', 'minority', 'majority', 'not minority', 'all'], help='perform undersampling to balance the dataset. only works with kfold')

    per_patient = subparsers.add_parser("per_patient")
    set_subparser_defaults(per_patient, per_patient_split)
    per_patient.add_argument("patient")
    per_patient.add_argument("--with-smote", action='store_true')
    per_patient.add_argument("--split-ratio", type=float, default=0.2)

    per_patient_time_split = subparsers.add_parser("per_patient_time_interval")
    set_subparser_defaults(per_patient_time_split, per_patient_time_interval)
    per_patient_time_split.add_argument("patient")
    per_patient_time_split.add_argument(
        "-t", "--train-time", required=True, type=int, help="split time in minutes"
    )

    select_n_async_cross = subparsers.add_parser("select_n_async_cross")
    set_subparser_defaults(select_n_async_cross, select_n_async_cross_patient_split)
    select_n_async_cross.add_argument("--n-samples", required=True, type=int)

    select_n_async_per_patient = subparsers.add_parser("select_n_async_per")
    set_subparser_defaults(select_n_async_per_patient, select_n_async_per_patient_split)
    select_n_async_per_patient.add_argument("--n-samples", required=True, type=int)
    select_n_async_per_patient.add_argument("patient")
    select_n_async_per_patient.add_argument("--with-smote", action='store_true')

    cross_patient_n_samples = subparsers.add_parser("cross_patient_n_samples")
    set_subparser_defaults(cross_patient_n_samples, cross_patient_n_samples_split)
    cross_patient_n_samples.add_argument("--n-samples", required=True, type=int)
    cross_patient_n_samples.add_argument("patient")
    cross_patient_n_samples.add_argument("--with-smote", action='store_true')

    train_on_everything_parser = subparsers.add_parser("train_on_all")
    set_subparser_defaults(train_on_everything_parser, train_on_everything_split)
    train_on_everything_parser.add_argument("-mf", "--model-file", default="model.pickle")
    train_on_everything_parser.set_defaults(train_on_all=True)
    train_on_everything_parser.add_argument("--with-smote", action='store_true')

    # regular parser arguments
    parser.add_argument("-b", "--bins", type=int, default=6)
    parser.add_argument("-w", "--write-results", action="store_true")
    parser.add_argument("--is-binning-analysis", action="store_true")

    parser.add_argument("-p", "--pickle-file")
    parser.add_argument("--feature-type", choices=feature_dict.keys())
    parser.add_argument("--async-type", choices=async_dict.keys())

    parser.add_argument("--run-chi2", action="store_true")
    parser.add_argument("-npf", "--new-pickling-file")
    parser.add_argument("-c", "--classifier", choices=CLASSIFIER_DICT.keys(), default="rf")
    parser.add_argument("--grid-search", action="store_true", help="Run the current model through a grid search of candidate hyperparameters in an attempt to better tune the model")
    parser.add_argument("--chi2-pruning", type=int, default=None)
    parser.add_argument("--n-clust", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("-o", "--outfile")
    parser.add_argument("--max-features", default=None, choices=["log2", "sqrt"])
    parser.add_argument("--n-estimators", default=10, type=int)
    parser.add_argument("--model-threshold", type=float)
    parser.add_argument("--damping", default=0.5, type=float)
    parser.add_argument("--affinity", choices=["precomputed", "euclidean"], default="euclidean")
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--cluster-all", type=bool, default=True)
    parser.add_argument("--linkage", choices=["ward", "complete", "average"], default="ward")
    parser.add_argument(
        "--examine-results",
        action="store_true",
        help="drop into IPython after results calculated and printed"
    )
    parser.add_argument(
        "--selected-features",
        nargs="+",
        help="manually select features to use in the classifier"
    )
    parser.add_argument(
        "--lda", action="store_true", help="perform linear discriminant analysis"
    )
    parser.add_argument(
        "--ica", type=int, help="perform independent component analysis"
    )
    parser.add_argument("--tsne", type=int, help="perform TSNE")
    parser.add_argument("-win", "--winsorize", type=float)
    parser.add_argument("-ncf", "--new-csv-file")
    # should I make the default robust?
    parser.add_argument("--scaler", default="min_max", choices=SCALER_DICT.keys())
    parser.add_argument("--pca", type=int)
    parser.add_argument("--cross-validate", action="store_true", help="Run the selected model with the selected hyperparams through cross validation in an attempt to derive a superior model")
    parser.add_argument("--rfecv", action="store_true", help="perform recursive feature elimination")
    parser.add_argument("--l1-selector", action="store_true", help="perform l1 feature selection")
    parser.add_argument("--write-chi2-results", action="store_true", help="write the results of the chi2 pruning to a file")
    parser.add_argument('--fdb1', action='store_true', help='first DTA is BSA/non-PVA, so reconvert prediction behind DTA to DTA regardless of the prediction')
    parser.add_argument('--fdb2', help='first DTA is a BSA/non-PVA, so reconvert prediction behind DTA to DTA if it is BSA', type=int)
    return parser


def additional_error_handling(args):
    """
    Because the creators of argparse seem to have checked out and are
    not including mutex groups
    """
    if args.pickle_file and (args.feature_type or args.async_type):
        raise Exception(
            "You cannot specify a pickle file along with a feature type or an "
            "async type!"
        )
    elif xor(bool(args.feature_type), bool(args.async_type)):
        raise Exception("You must specify both a feature and an asynchrony type!")
    elif not args.pickle_file and not args.feature_type and not args.async_type:
        raise Exception("You must specify a way to get data!")


def perform_pca(x_train, y_train, x_test, y_test, components):
    if components:
        p = PCA(n_components=components)
        p.fit(x_train, y_train)
        print "explained variance of {} components: {}".format(components, sum(p.explained_variance_ratio_))
        x_train = DataFrame(p.transform(x_train), index=y_train.index)
        x_test = DataFrame(p.transform(x_test), index=y_test.index)
    return x_train, x_test


class Winsorizor(object):
    def __init__(self, winsorize_val):
        self.val = winsorize_val
        self.col_max_mins = {}

    def fit_transform(self, x, to_transform=[]):
        if self.val and len(x) > 0:
            if to_transform:
                col_iter = to_transform
            else:
                col_iter = x.columns
            for col in col_iter:
                if col not in x.columns:
                    continue
                vals = winsorize(x[col], limits=self.val)
                self.col_max_mins[col] = {
                    'min': vals.min(),
                    'max': vals.max(),
                }
                x[col] = vals
        return x

    def transform(self, x):
        if self.val and len(x) > 0:
            for col, max_min in self.col_max_mins.iteritems():
                x.loc[x[x[col] > max_min['max']].index, col] = max_min['max']
                x.loc[x[x[col] < max_min['min']].index, col] = max_min['min']
        return x

    def to_pickle(self):
        pickle.dump(self, open("winsorizor.pickle", "wb"))


class ScalerWrapper(object):
    def __init__(self, scaler_type, preconfigured_scaler, x_train, x_test):
        if preconfigured_scaler:
            self.scaler = preconfigured_scaler
        elif scaler_type:
            self.scaler = SCALER_DICT[scaler_type]()
        if len(x_test) != 0:
            self.test_index = x_test.index
        else:
            self.test_index = []
        try:
            self.cols = x_test.columns
        except AttributeError:
            self.cols = x_train.columns
        self.x_train = x_train
        self.x_test = x_test

    def train_transform(self):
        if len(self.x_train) != 0:
            train_index = self.x_train.index
            x_train = self.scaler.fit_transform(self.x_train)
            return DataFrame(x_train, index=train_index, columns=self.cols)
        else:
            return self.x_train

    def test_transform(self):
        if len(self.x_test) != 0 and len(self.x_train) != 0:
            x_test = self.scaler.transform(self.x_test)
            return DataFrame(x_test, index=self.test_index, columns=self.cols)
        elif len(self.x_test) != 0:
            x_test = self.scaler.fit_transform(self.x_test)
            return DataFrame(x_test, index=self.test_index, columns=self.cols)
        else:
            return self.x_test

    def to_pickle(self):
        joblib.dump(self.scaler, "scaler.pickle")


def perform_space_replacement(df):
    col_mapping = {
        "I:E ratio": "I:E-ratio",
        "I:E ratio-prev": "I:E-ratio-prev",
        "I:E ratio-prev-prev": "I:E-ratio-prev-prev",
        "tve:tvi ratio-prev-prev": "tve:tvi-ratio-prev-prev",
        "tve:tvi ratio-prev": "tve:tvi-ratio-prev",
        "tve:tvi ratio": "tve:tvi-ratio",
    }
    return df.rename(columns=col_mapping)


def main():
    async_dict= {
        "dbl": get_binary_target_basic_dbl_trigger_data,
        "dbl_deriv_valid": get_binary_target_deriv_valid_dbl_trigger_data,
        "dbl_deriv_valid_retro": get_binary_target_retrospective_dbl_trigger_data,
        "fa": get_binary_target_fa_patient_data,
        "bs": get_binary_target_bs_patient_data,
        "bs_deriv_valid": get_binary_target_deriv_valid_bs_patient_data,
        "co": get_binary_target_cough_data,
        "su": get_binary_target_suction_data,
        "multi": get_multi_target_derivation_cohort_data,
        "multi_retro": get_multi_target_retrospective_data,
        "multi_retro_first_dbl_bs": get_multi_target_retrospective_data_first_dbl_as_bs,
        "multi_retro_fdb_deriv_val": get_multi_retrospective_data_fdb_deriv_val_cohort,
        "bs_dbl": get_bs_dbl_non_retrospective_data,
        "bs_dbl_retro": get_bs_dbl_retrospective_data,
        "bs_dbl_retro_fdb_deriv_val": get_bs_dbl_retrospective_first_dbl_as_bs_deriv_valid,
        "bs_dbl_retro_fdn_deriv_val": get_bs_dbl_retrospective_first_dbl_as_norm_deriv_valid,
        "bs_dbl_retro_fdb_deriv_cohort": get_bs_dbl_retrospective_data_first_dbl_as_bs,
        "bs_dbl_retro_fdb_val_cohort": get_bs_dbl_retrospective_data_first_dbl_as_bs_val_cohort,
        "multi_binary_retro": get_multi_class_binary_label_retrospective_data,
    }
    feature_dict = {
        "slope": get_slopes_of_pressure_curve_df,
        "integ": get_integrated_pressure_curve_df,
        "v1": get_v1,
        "v2": get_v2,
        "settings": get_vent_settings_features,
        "derived": get_derived_metadata_features,
        "metadata": get_settings_and_derived_features,
        "v2_and_metadata": get_v2_and_metadata,
        "fa_heuristic": get_fa_heuristic,
        "dbl_heuristic": get_dbl_trigger_heuristic_features,
        "dbl_all": get_dbl_trigger_heuristic_and_metadata_features,
        "dbl_chi2": get_dbl_chi2,
        "dbl_retro": get_dbl_trigger_retrospective_plus_metadata_features,
        "dbl_retro_chi2": get_dbl_retro_chi2,
        "dbl_no_retro_curated": get_dbl_curated,
        "bs_heuristic": get_bs_heuristic,
        "bs_all": get_bs_all,
        "bs_chi2": get_bs_chi2,
        "bs_curated": get_bs_curated,
        "co_heuristic": get_co_heuristic,
        "co_all": get_co_all,
        "co_curated": get_co_curated_no_tvi,
        "co_curated_with_tvi": get_co_curated_with_tvi,
        "su_heuristic": get_suction_heuristic,
        "su_all": get_suction_all,
        "su_curated": get_suction_curated,
        "retro_fused_plus_metadata": get_retro_plus_metadata,
        "greg_selection": get_all_greg_selected_features,
        "retro_non_noisy": get_retro_non_noisy,
        "retro_prev_plus_metadata": get_retro_prev_plus_metadata,
        "retro_prev_prev_plus_metadata": get_retro_prev_prev_plus_metadata,
        "retro_stripped_expert_plus_chi2": get_retro_stripped_expert_plus_chi2,
        "retro_stripped_expert_plus_chi2_2": get_retro_stripped_expert_plus_chi2_2,
        "retro_stripped_expert_plus_chi2_3": get_retro_stripped_expert_plus_chi2_3,
        "retro_stripped_expert_plus_chi2_4": get_retro_stripped_expert_plus_chi2_4,
        # this is currently highest performer
        "retro_stripped_expert_plus_chi2_5": get_retro_stripped_expert_plus_chi2_5,
        "retro_stripped_expert_plus_chi2_6": get_retro_stripped_expert_plus_chi2_6,
        "retro_stripped_expert_plus_chi2_7": get_retro_stripped_expert_plus_chi2_7,
        "retro_stripped_expert_plus_instrr": get_retro_stripped_expert_plus_instrr,
        "retro_stripped_expert_plus_instrr_prev": get_retro_stripped_expert_plus_instrr,
        "retro_stripped_low_prec": get_retro_stripped_lower_prec,
        "retro_stripped_high_prec": get_retro_stripped_higher_prec,
    }
    parser = build_parser(async_dict, feature_dict)
    args = parser.parse_args()
    additional_error_handling(args)
    feature_func = feature_dict.get(args.feature_type)
    gold_stnd_func = async_dict.get(args.async_type)

    x, y, extra_info = get_x_y(
        feature_func, args.bins, args.pickle_file, args.new_pickling_file, gold_stnd_func, args.new_csv_file
    )

    generator = args.split_func(x, y, args)
    results = []
    for x_train, x_test, y_train, y_test in generator:
        if args.only_patient:
            if args.only_patient not in x_test.patient.unique():
                continue
            else:
                x_test = x_test[x_test.patient == args.only_patient]
        try:
            del x_train['patient']
        except:
            pass
        try:
            del x_test['patient']
        except:
            pass

        x_train = perform_space_replacement(x_train)
        x_test = perform_space_replacement(x_test)
        if args.selected_features:
            x_train = x_train[args.selected_features]
            x_test = x_test[args.selected_features]

        # I guess I only wanted to winsorize two vars? Maybe I wanted to reduce
        # side effects
        winsorizor = Winsorizor(args.winsorize)
        x_train = winsorizor.fit_transform(x_train, ['tve:tvi-ratio', 'tve:tvi-ratio-prev'])
        x_test = winsorizor.transform(x_test)

        scaler = ScalerWrapper(args.scaler, None, x_train, x_test)
        classifier = Classifier(args, scaler)
        x_train = scaler.train_transform()
        x_test = scaler.test_transform()
        x_train, x_test = perform_pca(x_train, y_train, x_test, y_test, args.pca)
        if args.lda:
            lda = LinearDiscriminantAnalysis(solver='svd')
            cols = x_train.columns
            train_index, test_index = x_train.index, x_test.index
            lda.fit(x_train.values, y_train.values)
            x_train = DataFrame(lda.transform(x_train), index=train_index)
            x_test = DataFrame(lda.transform(x_test), index=test_index)

        if args.ica:
            fast_ica = FastICA(n_components=args.ica, whiten=True, random_state=True)
            train_index, test_index = x_train.index, x_test.index
            fast_ica.fit(x_train, y_train)
            x_train = DataFrame(fast_ica.transform(x_train), index=train_index)
            x_test = DataFrame(fast_ica.transform(x_test), index=test_index)

        if args.tsne:
            tsne = TSNE(n_components=args.tsne)
            train_index, test_index = x_train.index, x_test.index
            tsne.fit(x_train, y_train)
            x_train = tsne.fit_transform(x_train)
            import IPython; IPython.embed()

        # This was the best thing I could do with my given architecture.
        if (args.run_chi2 or args.chi2_pruning) and args.pca:
            raise ValueError("It doesn't make sense to run chi2 when using PCA!")
        if args.run_chi2:
            run_and_print_chi2(x_train, y_train)
        if args.chi2_pruning:
            patients = map(lambda x: x[1], x_test.index.str.split("-"))
            unique = set(patients)
            patient = "-".join(unique)
            x_train, x_test = perform_chi2_feature_pruning(x_train, x_test, y_train, args.chi2_pruning, args.write_chi2_results, args.pickle_file, patient)

        if not args.with_smote and (args.rfecv or args.l1_selector or args.grid_search):
            patient = map(lambda x: x[1], x_train.index.str.split("-"))
            x_train['patient'] = patient
            x_train.sort_values(by=['patient'], inplace=True)
            x_train = x_train.drop("patient", axis=1)

        if args.rfecv:
            x_train, x_test = classifier.backwards_feature_elimination(x_train, y_train, x_test)
        if args.l1_selector:
            x_train, x_test = classifier.l1_selection(x_train, y_train, x_test)

        if len(y_train) != 0:
            get_fa_elements(y_train, "training")
            get_fa_elements(y_test, "testing")
        else:
            get_fa_elements(y_test, "all")

        if len(x_train) != 0 and args.grid_search:
            classifier.grid_search(x_train, y_train)
        elif len(x_train) != 0 and args.cross_validate:
            classifier.cross_validate(x_train, y_train)
        elif len(x_train) != 0:
            classifier.fit(x_train, y_train)

        if args.train_on_all:
            classifier.write_to_file(args.model_file)
            scaler.to_pickle()
            winsorizor.to_pickle()
            return
        else:
            run_results = classifier.analyze_and_print_results(x_test, y_test, y_train, extra_info)
            results.append(run_results)

    if args.write_results:
        write_results(results, args)


if __name__ == "__main__":
    main()
