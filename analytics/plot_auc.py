from argparse import ArgumentParser
import csv
from itertools import cycle
import re

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import pylab as pyl
from tabulate import tabulate
import scipy
from sklearn.model_selection import learning_curve

from calc_averages import do_mean_calcs


def get_binary_results(files, x_type):
    """
    data structure should look like

    {
        <patient1>: {
            <x1>: [result1, result2, ...]
            <x2>: ...
        }
        <patient2>: ...
    }

    multiple results are assumed due to kfolding
    """
    ctrl_results = {}
    fa_results = {}
    for filename in files:
        with open(filename) as file_:
            split_file = filename.rstrip(".csv").split("-")
            try:
                regions = int(split_file[2])
            except ValueError:
                pass
            bin_val = float(split_file[3])
            patient = split_file[4]

            if patient not in ctrl_results:
                ctrl_results[patient] = dict()
            if patient not in fa_results:
                fa_results[patient] = dict()

            if "bin" == x_type:
                x_val = bin_val
            elif "regions" == x_type:
                x_val = regions

            reader = csv.reader(file_)
            for line in reader:
                if int(float(line[0])) == 0:
                    to_use = ctrl_results
                else:
                    to_use = fa_results

                if not x_val in to_use[patient]:
                    to_use[patient][x_val] = [line]
                else:
                    to_use[patient][x_val].append(line)
    return ctrl_results, fa_results


def perform_calc_averages(files, *args):
    """
    Gets classifier results with some kind of binning but for multiple
    different classes

    Take a classifier results file. They're usually structured like

    class,prec,sen,spec,fpr,f1,auc,fns,tns,fps,tps
    ...

    Then throw everything into a dataframe after processing the mean calculations
    for the file
    """
    rows = []
    columns = ['bin', 'async_class', 'f1', 'fpr', 'prec', 'sen', 'spec']
    for filename in files:
        split_file = filename.rstrip(".csv").split("-")
        bin_val = float(split_file[3])

        results = do_mean_calcs(filename)
        for class_ in results:
            vals = []
            for col in columns[2:]:
                vals.append(results[class_][col])
            rows.append([bin_val, class_] + vals)

    return pd.DataFrame(rows, columns=columns).sort_values(by=['bin', 'async_class'])


def get_cluster_results(files, x_type):
    results = {}
    for filename in files:
        with open(filename) as file_:
            split_file = filename.rstrip(".csv").split("-")
            regions = int(split_file[2])
            bin_val = float(split_file[3])
            patient = split_file[4]

            if patient not in results:
                results[patient] = dict()

            if "bin" == x_type:
                x_val = bin_val
            elif "regions" == x_type:
                x_val = regions

            reader = csv.reader(file_)
            for line in reader:
                if not line:
                    continue
                if not x_val in results[patient]:
                    results[patient][x_val] = [line]
                else:
                    results[patient][x_val].append(line)
    return results


def get_classifier_results(files, _):
    """
    Take a classifier results file. They're usually structured like

    class,prec,sen,spec,fpr,f1,auc,fns,tns,fps,tps
    ...

    Repeating over and over for each fold. In this case we just
    delegate everything out to calc_averages rather than rewrite
    all the logic

    Returns a DataFrame with columns that look like

    ['classifier', 'async_class', 'f1', 'fpr', 'prec', 'sen', 'spec']
    """
    rows = []
    columns = ['classifier', 'async_class', 'f1', 'fpr', 'prec', 'sen', 'spec']
    for file in files:
        match = re.search("results-(?P<cls>[a-z_]+)-", file)
        if not match:
            raise Exception("you messed up your regex or your filenames")
        cls = match.groupdict()['cls']
        results = do_mean_calcs(file)
        for class_ in results:
            vals = []
            for col in columns[2:]:
                vals.append(results[class_][col])
            rows.append([cls, class_] + vals)

    return pd.DataFrame(rows, columns=columns)


def process_cluster_results(results):
    processed = {'x': [], 'homogeneity': [], 'completeness': []}
    longest_x_patient_idx = np.argmax(map(lambda x: len(results[x]), results.keys()))
    longest_patient = results.keys()[longest_x_patient_idx]
    processed['x'] = sorted(results[longest_patient].keys())
    for val in processed['x']:
        homogeneity = []
        completeness = []
        for patient in results:
            try:
                for scores in results[patient][val]:
                    completeness.append(float(scores[2]))
                    homogeneity.append(float(scores[3]))
            except KeyError:
                pass
        processed['homogeneity'].append(sum(homogeneity) / float(len(homogeneity)))
        processed['completeness'].append(sum(completeness) / float(len(completeness)))
    return processed


def aggregate_plot_func(processed_results, x_lab, plot_title, legend_loc, png, binary_pva):
    x = sorted(processed_results.bin.unique())
    classes = processed_results.async_class.unique()
    color_map = {"DTA": ('r', 'c'), "BSA": ('m', 'g')}

    for idx, cls in enumerate(classes[classes > 0]):
        cls_df = processed_results[processed_results.async_class == cls]
        sens = []
        specs = []
        sen_err = []
        spec_err = []
        sen_color = color_map[binary_pva][0]
        spec_color = color_map[binary_pva][1]
        label = binary_pva
        sen_label = "{} sensitivity".format(label)
        spec_label = "{} specificity".format(label)

        for bin in x:
            bin_df = cls_df[cls_df.bin == bin]
            sens.append(bin_df.sen.mean())
            specs.append(bin_df.spec.mean())
            sen_err.append(bin_df.spec.std())
            spec_err.append(bin_df.spec.std())
        plt.errorbar(x, sens, yerr=sen_err, color=sen_color, label=sen_label)
        plt.errorbar(x, specs, yerr=spec_err, color=spec_color, label=spec_label)

    if len(x) > 20:
        rotation = -50
    else:
        rotation = 0
    plt.xticks(x, rotation=rotation)
    plt.yticks(np.arange(0, 1.1, .05))
    plt.ylim([0, 1.02])
    plt.xlim([min(x) - .1, max(x) + .1])
    plt.xlabel(x_lab)
    plt.title(plot_title)
    plt.ylabel("score")
    plt.legend(loc=legend_loc, borderaxespad=1.)
    plt.grid()
    if png:
        plt.savefig(png)
    else:
        plt.show()


def cluster_plot_func(processed_results, x_lab, plot_title, legend_loc, png, binary_pva):
    x = processed_results['x']
    plt.plot(x, processed_results['homogeneity'], marker='+', label='homogeneity')
    plt.plot(x, processed_results['completeness'], marker='o', label='completeness')
    plt.axis([min(x), max(x), 0, 1])
    plt.legend(loc='upper right', borderaxespad=1.)
    plt.ylabel("precision")
    plt.xlabel(x_lab)
    plt.title(plot_title)
    plt.show()


def multi_plot(processed_results, x_lab, plot_title, legend_loc, png, binary_pva):
    x = sorted(processed_results.bin.unique())
    classes = processed_results.async_class.unique()
    if len(classes) > 2:
        class_map = {1: "DTA", 2: "BSA"}
        color_map = {1: ('r', 'c'), 2: ('m', 'g')}

    for idx, cls in enumerate(classes[classes >= 1]):
        cls_df = processed_results[processed_results.async_class == cls]
        sens = []
        specs = []
        sen_err = []
        spec_err = []
        sen_color = color_map[cls][0]
        spec_color = color_map[cls][1]

        if len(classes) > 2:
            label = class_map[cls]
            sen_label = "{} sensitivity".format(label)
            spec_label = "{} specificity".format(label)
        else:
            sen_label = "sensitivity"
            spec_label = "specificity"

        for bin in x:
            bin_df = cls_df[cls_df.bin == bin]
            sens.append(bin_df.sen.mean())
            specs.append(bin_df.spec.mean())
            sen_err.append(bin_df.spec.std())
            spec_err.append(bin_df.spec.std())
        plt.errorbar(x, sens, yerr=sen_err, color=sen_color, label=sen_label)
        plt.errorbar(x, specs, yerr=spec_err, color=spec_color, label=spec_label)

    if len(x) > 20:
        rotation = -50
    else:
        rotation = 0
    plt.xticks(x, rotation=rotation)
    plt.yticks(np.arange(0, 1.1, .05))
    plt.ylim([0, 1.02])
    plt.xlim([min(x) - .1, max(x) + .1])
    plt.xlabel(x_lab)
    plt.title(plot_title)
    plt.ylabel("score")
    plt.legend(loc=legend_loc, borderaxespad=1.)
    plt.grid()
    if png:
        plt.savefig(png)
    else:
        plt.show()


def classifier_diff_plot(processed_results, x_lab, plot_title, legend_loc, png, binary_pva):
    x = processed_results.classifier.unique()
    classes = processed_results.async_class.unique()
    xticks_map = {"nn": "MLP", "etc": "ERTC", "gbrt": "GBC", "rf": "RF", "ovr_rf": "OVR", "vote": "Static Voting", "cust_vote": "Custom Ensemble", "soft_vote": "Ensemble"}
    if binary_pva:
        class_map = {0: "Non-PVA", 1: binary_pva}
        if binary_pva == "BSA":
            colors = ('m', 'g')
        elif binary_pva == 'DBLA' or binary_pva == 'DTA':
            colors = ('r', 'c')
        color_map = {0: ('aqua', 'orange'), 1: colors}
    elif len(classes) > 2:
        class_map = {0: "Non-PVA", 1: "DTA", 2: "BSA"}
        color_map = {0: ('aqua', 'orange'), 1: ('r', 'c'), 2: ('m', 'g')}

    n_classes = len(classes[classes >= 0])
    width = (1.0 - .2) / (n_classes * 2)
    for idx, cls in enumerate(classes):
        sen = processed_results[processed_results.async_class == cls].sen.values
        spec = processed_results[processed_results.async_class == cls].spec.values
        sen_color = color_map[cls][0]
        spec_color = color_map[cls][1]
        label = class_map[cls]
        sen_label = "{} sensitivity".format(label)
        spec_label = "{} specificity".format(label)
        plt.bar(np.arange(len(x)) + (width * (idx*2)), sen, width, color=sen_color, label=sen_label)
        plt.bar(np.arange(len(x)) + (width * ((idx*2)+1)), spec, width , color=spec_color, label=spec_label)

    xticks = [xticks_map[key] for key in x]
    plt.xticks(np.arange(len(x)) + width*n_classes, xticks)
    plt.yticks(np.arange(0, 1.1, .05))
    plt.ylim([0, 1.02])
    plt.xlabel("")
    plt.title(plot_title)
    plt.ylabel("score")
    plt.legend(loc=legend_loc, framealpha=0.8)
    plt.grid()
    if png:
        plt.savefig(png)
    else:
        plt.show()


def show_binary_table(x_lab, processed):
    print tabulate(processed, headers='keys')


def main():
    parser = ArgumentParser()
    parser.add_argument("--x-type", choices=["bin", "regions"], required=True)
    parser.add_argument("--x-lab", required=True)
    parser.add_argument("--legend-loc", default='lower left')
    parser.add_argument("--title", required=True)
    parser.add_argument("--png", help="Filename to save the graph to")
    subparsers = parser.add_subparsers()
    aggregate_plotter = subparsers.add_parser("aggregate")
    aggregate_plotter.set_defaults(
        plot_func=aggregate_plot_func,
        process_func=lambda x: x,
        results_func=perform_calc_averages,
        table_func=show_binary_table,
        binary_pva=None,
    )
    aggregate_plotter.add_argument("--binary-pva", choices=['DTA', 'DBLA', 'BSA'], required=True)
    aggregate_plotter.add_argument("files", nargs="*")
    cluster_plotter = subparsers.add_parser("clustering")
    cluster_plotter.set_defaults(
        plot_func=cluster_plot_func,
        process_func=process_cluster_results,
        results_func=get_cluster_results,
        table_func=lambda x, y: x,  # don't care
        binary_pva=None
    )
    cluster_plotter.add_argument("files", nargs="*")
    classifier_diff = subparsers.add_parser("classifier")
    classifier_diff.add_argument("--binary-pva", choices=['DTA', 'DBLA', 'BSA'])
    classifier_diff.set_defaults(
        results_func=get_classifier_results,
        process_func=lambda x: x,
        table_func=lambda x,y: x, # XXX TODO
        plot_func=classifier_diff_plot,
    )
    classifier_diff.add_argument("files", nargs="*")
    multi = subparsers.add_parser("multi")
    multi.set_defaults(
        results_func=perform_calc_averages,
        process_func=lambda x: x,
        table_func=lambda x,y: x, # XXX TODO
        plot_func=multi_plot,
        binary_pva=None
    )
    multi.add_argument("files", nargs="*")
    args = parser.parse_args()

    results = args.results_func(args.files, args.x_type)
    processed = args.process_func(results)
    args.table_func(args.x_lab, processed)
    args.plot_func(processed, args.x_lab, args.title, args.legend_loc, args.png, args.binary_pva)


if __name__ == "__main__":
    main()
