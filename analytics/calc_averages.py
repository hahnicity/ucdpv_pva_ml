from argparse import ArgumentParser
from copy import deepcopy
import csv

from prettytable import PrettyTable


def do_mean_calcs(file):
    inner_results = {
        "fns": [], "tns": [], "fps": [], "tps": [],
    }
    inter_results = {i: deepcopy(inner_results) for i in range(5)}
    reader = csv.reader(open(file))
    mapping = {-4: "fns", -3: "tns", -2: "fps", -1: "tps"}
    for idx, line in enumerate(reader):
        for i in range(-4, 0):
            inter_results[int(float(line[0]))][mapping[i]].append(float(line[i]))
    for k, v in inter_results.items():
        if v == inner_results:
            del inter_results[k]
    final_results = {i: {} for i in inter_results.keys()}
    for label, v in inter_results.items():
        try:
            final_results[label]["prec"] = round(sum(v["tps"]) / sum(v["tps"] + v["fps"]), 4)
        except ZeroDivisionError:
            # If its 0 there are no tps+fps. This means we just didn't predict
            # any positives
            final_results[label]['prec'] = 0.0
        final_results[label]['acc'] = round(
             (sum(v['tps']) + sum(v['tns'])) /
             (sum(v['tps']) + sum(v['tns']) + sum(v['fps']) + sum(v['fns'])), 4
        )
        final_results[label]["sen"] = round(sum(v["tps"]) / sum(v["tps"] + v["fns"]), 4)
        final_results[label]["spec"] = round(sum(v["tns"]) / sum(v["tns"] + v["fps"]), 4)
        final_results[label]["fpr"] = round(sum(v["fps"]) / sum(v["tns"] + v["fps"]), 4)
        try:
            final_results[label]["npv"] = round(sum(v["tns"]) / sum(v["tns"] + v["fns"]), 4)
        except:
            final_results[label]['npv'] = 0.0
        try:
            final_results[label]["f1"] = round(2 * (
                final_results[label]["prec"] * final_results[label]["sen"]
            ) / (
                final_results[label]["prec"] + final_results[label]["sen"]
            ), 4)
        except ZeroDivisionError:
            final_results[label]['f1'] = 0.0
    return final_results


def do_prettytable(results):
    pt = PrettyTable()
    order = ['acc', "prec", "sen", "spec", "npv", "fpr", "f1"]
    pt.field_names = [
        "class", 'accuracy', "precision", "sensitivity", "specificity", "npv", "fpr", "f1-score"
    ]
    for class_ in results.keys():
        vals = [results[class_][order[i]] for i in range(len(order))]
        pt.add_row([class_] + vals)
    print(pt)


def main():
    parser = ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    print "filename is: {}".format(args.file)
    results = do_mean_calcs(args.file)
    do_prettytable(results)


if __name__ == "__main__":
    main()
