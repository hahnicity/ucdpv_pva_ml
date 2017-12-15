from math import ceil, floor
import pickle

import IPython
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_pickle, Series
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.cluster import (
    AgglomerativeClustering,
    AffinityPropagation,
    DBSCAN,
    KMeans,
    MeanShift,
    SpectralClustering
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.externals import joblib
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


class CustomMultiClassVoteClassifier(object):
    def __init__(self, estimators):
        self.classifiers = estimators

    def fit(self, x, y):
        for cls in self.classifiers:
            cls[1].fit(x, y)

    def predict(self, x):
        """
        Allow the ETC to be solely responsible for predicting DBLA
        and then allow the other predictors to come to a conclusion on BSA
        the majority vote on BSA will win.
        """
        cls_results = 1
        bs_predictors = []
        for cls in self.classifiers:
            predictions = Series(cls[1].predict(x))
            if cls[0] == "etc":
                predictions.loc[predictions[predictions == 1]] = 0
                bs_predictors.append(predictions)
            else:
                predictions.loc[predictions[predictions == 2]] = 0
                dbl_predictor = predictions
        bs_sum = sum(bs_predictors)
        bs_idx = bs_sum[bs_sum == 2 * len(bs_predictors)].index
        dbl_predictor.loc[bs_idx] = 2
        return list(dbl_predictor)


CLASSIFIER_DICT = {
    "ada": AdaBoostClassifier,
    "agg": AgglomerativeClustering,
    "gbrt": GradientBoostingClassifier,
    "means": MeanShift,
    "aff": AffinityPropagation,
    "km": KMeans,
    "svm": SVC,
    "rf": RandomForestClassifier,
    "ovr_rf": RandomForestClassifier,
    "ovo_rf": RandomForestClassifier,
    "nn": MLPClassifier,
    "dbs": DBSCAN,
    "gnb": GaussianNB,
    "dtc": DecisionTreeClassifier,
    "etc": ExtraTreesClassifier,
    "spec": SpectralClustering,
    "vote": VotingClassifier,
    "cust_vote": CustomMultiClassVoteClassifier,
    "soft_vote": VotingClassifier,
}


def get_fns_idx(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return pos_loc[pos_loc != label].index


def get_fns(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc != label])


def get_tns(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc != label])


def get_fps_idx(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return neg_loc[neg_loc == label].index


def get_fps_full_rows(actual, predictions, label, filename):
    idx = get_fps_idx(actual, predictions, label)
    full_df = read_pickle(filename)
    return full_df.loc[idx]


def get_fps(actual, predictions, label):
    neg = actual[actual != label]
    neg_loc = predictions.loc[neg.index]
    return len(neg_loc[neg_loc == label])


def get_tps(actual, predictions, label):
    pos = actual[actual == label]
    pos_loc = predictions.loc[pos.index]
    return len(pos_loc[pos_loc == label])


def false_positive_rate(actual, predictions, label):
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 0
    else:
        return float(fp) / (fp + tn)


def specificity(actual, predictions, label):
    """
    Also known as the true negative rate
    """
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 1
    else:
        return float(tn) / (tn + fp)


def dbscan_cross_validation(x_train, y_train):
    pass


def kmeans_cross_validation(x_train, y_train):
    pass


def nn_cross_validation(x_train, y_train):
    params = {
        "hidden_layer_sizes": [
            (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 50),
            (10, 10, 10), (20, 20, 20), (20, 10, 20), (50, 50, 50), (50, 10, 50),
            (100, 20, 50), (100, 50, 20), (100, 50, 50)
        ],
        "activation": ["tanh"],
        #"solver": ["sgd", "adam", "lbfgs"],
    }
    cv = KFold(n_splits=10)
    clf = GridSearchCV(MLPClassifier(solver="lbfgs", random_state=1), params, cv=cv)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def etc_cross_validation(x_train, y_train):
    # ETC can perform very well on DBL w/ recall when max_depth is set very low
    # @ max_depth of 2, I got sen=1
    params = {
        "n_estimators": np.arange(30, 100, 10),
        "criterion": ["gini"],
        "max_depth": [3],
        "min_samples_leaf": [1, 2, 3],
    }
    cv = KFold(n_splits=10)
    clf = GridSearchCV(ExtraTreesClassifier(random_state=1), params, cv=cv)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def gbrt_cross_validation(x_train, y_train):
    # cross validation seems to imply the following params are best, but
    # when running everything through testing performance actually decreased.
    #
    #"gbrt": {"random_state": 1, "learning_rate": 0.05, "max_depth": 5, "n_estimators": 200},
    params = {
        "learning_rate": [0.08, 0.05, .02],
        "max_depth": [5],
        "n_estimators": [200, 300, 400],
        "criterion": ['friedman_mse'],
        "max_leaf_nodes": [40, 50, 60],
    }
    cv = KFold(n_splits=10)
    clf = GridSearchCV(GradientBoostingClassifier(random_state=1), params, cv=cv)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def rf_cross_validation(x_train, y_train):
    params = {
        "n_estimators": np.arange(10, 40, 10),
        "criterion": ["entropy"],
        "min_impurity_split": [1e-6, 1e-2],
        "min_samples_leaf": [1, 2, 3],
    }
    cv = KFold(n_splits=10)
    clf = GridSearchCV(RandomForestClassifier(random_state=1), params, cv=cv)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def svm_cross_validation(x_train, y_train):
    params = {
        "C": [.01, .1, 1, 5, 10, 20, 50],
        "gamma": [.001, .01, .1, 1, 2, 5],
    }
    cv = ShuffleSplit(train_size=.8, test_size=.2, n_splits=10)
    clf = GridSearchCV(SVC(random_state=1), params, cv=cv)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


class FakeFeatureSelector(object):
    def fit(self, x, y):
        pass

    def transform(self, x):
        return x


class Classifier(object):
    # input the scaler so we can do reverse transformations for
    # debugging
    def __init__(self, args, scaler):
        self.scaler = scaler
        rf_params = {
            "random_state": 1,
            "criterion": "entropy",
            "max_features": args.max_features,
            "max_depth": 15,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": 2,
            "min_impurity_split": 1e-6,
        }
        gbrt_params = {"random_state": 1}
        nn_params = {
            "solver": "lbfgs",
            "activation": "tanh",
            "random_state": 1,
            "hidden_layer_sizes": (100, 20, 50),
        }
        etc_params = {
            "n_estimators": 50,
            "criterion": "gini",
            "max_depth": 2,
            "min_samples_leaf": 4,
        }
        self.hyperparams = {
            "ada": {"n_estimators": args.n_estimators, "random_state": 1},
            "agg": {"n_clusters": args.n_clust, "linkage": args.linkage},
            "aff": {"damping": args.damping, "affinity": args.affinity},
            # 700 seems to be best for optimizing on sensitivity. I actually
            # got to .97/.97 with .1 winsorizor here. precision is pretty low tho.
            # but technically all i need to report is sen/spec in med journals.
            #
            # maybe its the .01 winsorizor because regardless of hidden count
            # from 200 on up im gettin >.96
            "gbrt": gbrt_params,
            "means": {"bandwidth": args.bandwidth, "cluster_all": args.cluster_all},
            "spec": {"n_clusters": args.n_clust},
            "km": {"n_clusters": args.n_clust},
            # nn performance seemed to increase 1% when we added more neurons and
            # an additional layer to (50,50,50). What about 4+ layers??
            "nn": nn_params,
            "rf": rf_params,
            "ovr_rf": rf_params,
            "ovo_rf": rf_params,
            "dbs": {"eps": args.eps},
            "dtc": {"random_state": 1},
            "etc": etc_params,
            "vote": {
                "estimators": [
                    #("rf", RandomForestClassifier(**rf_params)),
                    ("gbrt", GradientBoostingClassifier(**gbrt_params)),
                    ("etc", ExtraTreesClassifier(**etc_params)),
                    ("nn", MLPClassifier(**nn_params)),
                ],
                "weights": [4, 7, 4],
                "voting": "hard",
            },
            "soft_vote": {
                "estimators": [
                    ("gbrt", GradientBoostingClassifier(**gbrt_params)),
                    ("gbrt2", GradientBoostingClassifier(n_estimators=150, random_state=1)),
                    ("gbrt3", GradientBoostingClassifier(n_estimators=200, random_state=1, max_depth=2)),
                    ("gbrt4", GradientBoostingClassifier(n_estimators=250, random_state=1, max_depth=2)),
                    ("gbrt5", GradientBoostingClassifier(n_estimators=150, random_state=1, max_depth=2)),
                    ("gbrt6", GradientBoostingClassifier(n_estimators=120, random_state=1, max_depth=2)),
                    ("etc", ExtraTreesClassifier(**etc_params)),
                    ("nn", MLPClassifier(**nn_params)),
                    ("nn2", MLPClassifier(**{
                        "solver": "lbfgs",
                        "activation": "tanh",
                        "random_state": 1,
                        "hidden_layer_sizes": (20, 20),
                    })),
                    ("nn3", MLPClassifier(**{
                        "solver": "lbfgs",
                        "activation": "tanh",
                        "random_state": 1,
                        "hidden_layer_sizes": (10, 10, 10),
                    })),
                ],
                "voting": "soft",
            },
            "cust_vote": {
                "estimators": [
                    ("rf", RandomForestClassifier(**rf_params)),
                    ("gbrt", GradientBoostingClassifier(**gbrt_params)),
                    ("nn", MLPClassifier(**nn_params)),
                ],
            },
        }
        model_hyperparams = self.hyperparams.get(args.classifier, {})
        self.classifier = CLASSIFIER_DICT[args.classifier](**model_hyperparams)
        self.cv_func = {
            "svm": svm_cross_validation,
            "rf": rf_cross_validation,
            "nn": nn_cross_validation,
            "gbrt": gbrt_cross_validation,
            "etc": etc_cross_validation,
        }.get(args.classifier, lambda x, y: None)
        wrapper = {
            "ovr_rf": OneVsRestClassifier,
            "ovo_rf": OneVsOneClassifier,
        }.get(args.classifier, lambda x: x)
        self.classifier = wrapper(self.classifier)
        if args.model_threshold:
            self.feature_selector = SelectFromModel(
                RandomForestClassifier(max_features=None, random_state=1),
                threshold=args.model_threshold
            )
        else:
            self.feature_selector = FakeFeatureSelector()
        self.examine_results = args.examine_results
        self.pickle_file = args.pickle_file
        self.reclassify_first_dta_regardless = args.fdb1
        self.reclassify_first_dta_on_cond = args.fdb2

    def _is_clustering(self):
        if type(self.classifier) in [
            KMeans, DBSCAN, SpectralClustering, AffinityPropagation, MeanShift, AgglomerativeClustering
        ]:
            return True
        else:
            return False

    def analyze_and_print_results(self, x_test, y_test, y_train, extra_info):
        if len(y_train) == 0:
            predictions = self.fit_predict(x_test)
        else:
            predictions = self.predict(x_test)
        predictions = Series(predictions, index=y_test.index)
        if self.reclassify_first_dta_regardless:
            existing_index = y_test.index
            predictions.index = range(len(predictions))
            y_test.index = range(len(predictions))
            dta_obs = y_test[y_test == 1].index
            if len(dta_obs) != 0:
                y_test.loc[dta_obs - 1] = 1
            dta_predictions = predictions[predictions == 1].index
            if len(dta_predictions) != 0:
                predictions.loc[(dta_predictions - 1).difference([-1])] = 1
            y_test.index = existing_index
            predictions.index = existing_index
        elif self.reclassify_first_dta_on_cond is not None:
            existing_index = y_test.index
            predictions.index = range(len(predictions))
            y_test.index = range(len(predictions))
            dta_obs = y_test[y_test == 1].index
            if len(dta_obs) != 0:
                y_test.loc[dta_obs - 1] = 1
            dta_predictions = predictions[predictions == 1].index
            if len(y_test.unique()) > 2:
                first_breath = predictions.loc[dta_predictions - 1]
                dta_first_breath = first_breath[first_breath == self.reclassify_first_dta_on_cond].index.difference([-1])
            else:
                dta_first_breath = (dta_predictions - 1).difference([-1])
            if len(dta_first_breath) > 0:
                predictions.loc[dta_first_breath] = 1
            y_test.index = existing_index
            predictions.index = existing_index
        if self._is_clustering():
            results = self.get_overall_clustering_results(y_test, predictions)
            self.print_clustering_results(results)
            self.print_per_cluster_results(y_test, predictions)
        else:
            results = self.get_non_clustering_results(y_test, predictions)
            self.print_non_clustering_results(results)
        if self.examine_results:
            IPython.embed()
        return [results]

    def cross_validate(self, x_train, y_train):
        cv = KFold(n_splits=10)
        clf = GridSearchCV(self.classifier, {}, cv=cv)
        clf.fit(x_train, y_train)
        self.classifier = clf.best_estimator_

    def fit(self, x_train, y_train):
        self.feature_selector.fit(x_train, y_train)
        row1 = x_train.iloc[0]
        x_train = self.feature_selector.transform(x_train)
        if isinstance(self.feature_selector, SelectFromModel):
            print self._find_which_features_used(row1, x_train[0])
        self.print_features_remaining(x_train)
        self.x_train, self.y_train = x_train, y_train
        self.classifier.fit(x_train, y_train)

    def _find_which_features_used(self, original_row1, new_row1):
        features = []
        for val in new_row1:
            for col, old_val in original_row1.iteritems():
                if old_val == val:
                    features.append(col)
        return features

    def fit_predict(self, x_test):
        return self.classifier.fit_predict(x_test)

    def grid_search(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.classifier = self.cv_func(x_train, y_train)


    def get_overall_clustering_results(self, actual, predictions):
        ami = metrics.adjusted_mutual_info_score(actual, predictions)
        ars = metrics.adjusted_rand_score(actual, predictions)
        completeness = metrics.completeness_score(actual, predictions)
        homogeneity = metrics.homogeneity_score(actual, predictions)
        v_measure = metrics.v_measure_score(actual, predictions)
        return [ami, ars, completeness, homogeneity, v_measure]

    def get_non_clustering_results(self, actual, predictions):
        rows = []
        for label in sorted(self.y_train.unique()):
            prec = round(metrics.precision_score(
                actual, predictions, average=None, labels=[label])[0], 4
            )
            sen = round(metrics.recall_score(
                actual, predictions, average=None, labels=[label])[0], 4
            )
            spec = round(specificity(actual, predictions, label), 4)
            f1 = round(metrics.f1_score(
                actual, predictions, average=None, labels=[label])[0], 4
            )
            roc = metrics.roc_curve(actual, predictions, pos_label=label)
            auc = round(metrics.auc(roc[0], roc[1]), 4)
            fpr = round(false_positive_rate(actual, predictions, label), 4)

            fps = get_fps(actual, predictions, label)
            tps = get_tps(actual, predictions, label)
            fns = get_fns(actual, predictions, label)
            tns = get_tns(actual, predictions, label)
            row = [label, prec, sen, spec, fpr, f1, auc, fns, tns, fps, tps]
            rows.append(row)
        return rows

    def l1_selection(self, x_train, y_train, x_test):
        cols = x_train.columns
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(x_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)
        x_train = model.transform(x_train)
        print("Features selected: {}".format(list(cols[model.get_support()])))
        x_train = DataFrame(x_train, columns=cols[model.get_support()])
        x_test = DataFrame(x_test, columns=cols[model.get_support()])
        return x_train, x_test

    def backwards_feature_elimination(self, x_train, y_train, x_test):
        cv = KFold(n_splits=10)
        rfecv = RFECV(self.classifier, cv=cv)
        rfecv.fit(x_train, y_train)
        cols = x_train.columns[rfecv.support_]
        print("Features selected: {}".format(list(cols)))
        return x_train.loc[:,cols], x_test.loc[:,cols]

    def predict(self, x_test):
        x_test = self.feature_selector.transform(x_test)
        return self.classifier.predict(x_test)

    def print_clustering_results(self, scores):
        table = PrettyTable()
        table.field_names = ["AMI", "ARI", "Completeness", "Homoegeneity", "V-Measure"]
        table.add_row(scores)
        print(table)

    def print_features_remaining(self, x):
        if isinstance(self.feature_selector, SelectFromModel):
            print "Features after model pruning: {}".format(x.shape[1])

    def print_non_clustering_results(self, results):
        table = PrettyTable()
        table.field_names = [
            "class", "precision", "sensitivity", "specificity", "fpr", "f1-score", "AUC"
        ]
        for row in results:
            table.add_row(row[:-4])
        print(table)

    def print_per_cluster_results(self, actual, predictions):
        rows = []
        for cluster in sorted(predictions.unique()):
            corresponding = actual[predictions[predictions == cluster].index]
            row = [cluster]
            for real in sorted(actual.unique()):
                row.append(float(len(corresponding[corresponding == real])))
            rows.append(row)
        table = PrettyTable()
        field_names = ["cluster"] + [
            "{}-{}".format(i, type_)
            for i in sorted(actual.unique())
            for type_ in ["count"]
        ]
        table.field_names = field_names
        for row in rows:
            table.add_row(row)
        print(table)

    def inverse_transform(self, x):
        """
        A debugging method to be used when you want to reverse a scaler's
        transformation
        """
        cols = x.columns
        index = x.index
        trans = self.scaler.scaler.inverse_transform(x)
        return DataFrame(trans, columns=cols, index=index)

    def tsne_visualize_classifier_diff(self, x_test, y_test, predictions, obs):
        """
        Method used for the paper to display how class boundary points for
        ETC are more conducive to DBL detection.

        This method only works with the 3-dimensional expert feature set.
        """
        models = [
            CLASSIFIER_DICT['etc'](**self.hyperparams['etc']),
            CLASSIFIER_DICT['gbrt'](**self.hyperparams['gbrt']),
            CLASSIFIER_DICT['nn'](**self.hyperparams['nn']),
            CLASSIFIER_DICT['soft_vote'](**self.hyperparams['soft_vote']),
        ]
        train_index = self.x_train.iloc[0:obs].index
        x = self.x_train.loc[train_index].append(x_test)
        y = self.y_train.loc[train_index].append(y_test)
        tsne = TSNE(n_components=2)
        index = x.index
        tsne.fit(x, y)
        x = DataFrame(tsne.fit_transform(x), index=index)
        x_test = x.loc[x_test.index]
        y_test = y.loc[x_test.index]

        for model in models:
            model.fit(x.loc[train_index], y.loc[train_index])

        plot_idx = 1
        plot_step = .1
        plt.subplots(1, 4, sharex=True, sharey=True)
        for model in models:
            y = y_test

            model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]

            plt.subplot(1, 4, plot_idx)
            if plot_idx <= len(models):
                if model_title.strip() == "ExtraTrees":
                    model_title = "ERTC"
                elif model_title.strip() == "GradientBoosting":
                    model_title = "GBC"
                elif model_title.strip() == "Voting":
                    model_title = "Ensemble"
                plt.title(model_title)

            col1 = x.iloc[:,0]
            col2 = x.iloc[:,1]
            x_min, x_max = col1.min() - 1, col1.max() + 1
            y_min, y_max = col2.min() - 1, col2.max() + 1
            #plt.xticks(np.arange(floor(x_min), ceil(x_max + 2.0)))
            # only works with expert feature set
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, plot_step),
                np.arange(y_min, y_max, plot_step),
            )

            grid = np.c_[xx.ravel(), yy.ravel()]
            contourxx = xx
            contouryy = yy
            plot_colors = ['aqua', 'r', 'm']
            cmap = LinearSegmentedColormap.from_list('foo', colors=['aqua', 'r', 'm'])
            if isinstance(model, MLPClassifier) or isinstance(model, GradientBoostingClassifier):
                Z = model.predict(grid)
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(contourxx, contouryy, Z, cmap=cmap)
            else:
                estimator_alpha = 1.0 / len(model.estimators_)
                Z = model.predict(grid)
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(contourxx, contouryy, Z, alpha=estimator_alpha, cmap=cmap)
                cs = plt.contourf(contourxx, contouryy, Z, cmap=cmap)

            n_classes = len(self.y_train.unique())
            target_names = {0: "Non-PVA", 1: "DTA", 2: "BSA"}
            for i, c in zip(xrange(n_classes), plot_colors):
                idx = np.where(y_test == i)
                plt.scatter(x_test.values[idx, 0], x_test.values[idx, 1], c=c, label=target_names[i], cmap=cmap)

            plt.tick_params(labelright='off', right='off')
            plot_idx += 1

        plt.suptitle("Decision Boundary Plot for Classifiers")
        plt.axis("tight")
        plt.show()

    def visualize_classifier_diff(self, cols, x_test, y_test, predictions):
        """
        Method used for the paper to display how class boundary points for
        ETC are more conducive to DBL detection.

        This method only works with the 3-dimensional expert feature set.
        """
        models = [
            CLASSIFIER_DICT['etc'](**self.hyperparams['etc']),
            CLASSIFIER_DICT['gbrt'](**self.hyperparams['gbrt']),
            CLASSIFIER_DICT['nn'](**self.hyperparams['nn']),
            CLASSIFIER_DICT['soft_vote'](**self.hyperparams['soft_vote']),
        ]
        for model in models:
            model.fit(self.x_train[cols], self.y_train)

        plot_idx = 1
        plot_step = 0.02
        plt.subplots(1, 4, sharex=True, sharey=True)
        for model in models:
            X = x_test[cols]
            y = y_test

            model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]

            plt.subplot(1, 4, plot_idx)
            if plot_idx <= len(models):
                if model_title.strip() == "ExtraTrees":
                    model_title = "ERTC"
                elif model_title.strip() == "GradientBoosting":
                    model_title = "GBC"
                elif model_title.strip() == "Voting":
                    model_title = "Ensemble"
                plt.title(model_title)

            col1 = X[cols[0]]
            col2 = X[cols[1]]
            x_min, x_max = col1.min() - 1, col1.max() + 1
            y_min, y_max = col2.min() - 1, col2.max() + 1
            plt.xticks(np.arange(floor(x_min), ceil(x_max + 2.0)))
            # only works with expert feature set
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, plot_step),
                np.arange(y_min, y_max, plot_step),
            )

            grid = np.c_[xx.ravel(), yy.ravel()]
            contourxx = xx
            contouryy = yy
            plot_colors = ['aqua', 'r', 'm']
            cmap = LinearSegmentedColormap.from_list('foo', colors=['aqua', 'r', 'm'])
            if isinstance(model, MLPClassifier) or isinstance(model, GradientBoostingClassifier):
                Z = model.predict(grid)
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(contourxx, contouryy, Z, cmap=cmap)
            else:
                estimator_alpha = 1.0 / len(model.estimators_)
                Z = model.predict(grid)
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(contourxx, contouryy, Z, alpha=estimator_alpha, cmap=cmap)
                cs = plt.contourf(contourxx, contouryy, Z, cmap=cmap)

            n_classes = len(self.y_train.unique())
            target_names = {0: "Non-PVA", 1: "DTA", 2: "BSA"}
            for i, c in zip(xrange(n_classes), plot_colors):
                idx = np.where(y_test == i)
                plt.scatter(X.values[idx, 0], X.values[idx, 1], c=c, label=target_names[i], cmap=cmap)

            name_map = {
                "tve:tvi-ratio": "TVe/TVi",
                "tve:tvi-ratio-prev": "TVe/TVi-previous",
                "eTime-prev": "eTime-previous"
            }
            plt.xlabel(name_map[cols[0]])
            if plot_idx == 1:
                plt.ylabel(name_map[cols[1]])
            plt.tick_params(labelright='off', right='off')
            plot_idx += 1

        plt.suptitle("Decision Boundary Plot for Classifiers")
        plt.axis("tight")
        plt.show()

    def visualize_all(self, cols, x_test, y_test, predictions):
        """
        Visualize the results of your algorithm with respect to all predictions
        made
        """
        x_test = self.inverse_transform(x_test)
        self._visualize(cols, x_test, y_test, predictions, predictions)

    def visualize_label_results(self, cols, x_test, y_test, predictions, label):
        """
        Visualize the results of your algorithm in a 2D scatter plot by picking
        two columns to serve as x and y. Then the predictions will show up as
        colors
        """
        x_test = self.inverse_transform(x_test)
        zone = predictions.loc[y_test[y_test == label].index]
        self._visualize(cols, x_test, y_test, predictions, zone)

    def _visualize(self, cols, x_test, y_test, predictions, zone):
        x = x_test.loc[zone.index,cols[0]]
        y = x_test.loc[zone.index,cols[1]]
        color_mapping = {0: 'b', 1: 'orange', 2: 'g', 3: 'm', 4: 'c'}
        if len(cols) != 2:
            raise Exception("you can only pick 2 cols to use")
        colors = []
        for class_ in list(zone):
            colors.append(color_mapping[int(class_)])

        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.ylim([y.min()-0.04, y.max()+0.04])
        plt.xlim([x.min()-0.04, x.max()+0.04])
        plt.scatter(x, y, c=colors)
        plt.show()

    def write_to_file(self, filename):
        joblib.dump(self.classifier, filename)
