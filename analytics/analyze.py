from math import sqrt

from prettytable import PrettyTable
from sklearn.feature_selection import chi2


def run_chi2(x, y):
    x2, p_vals = chi2(x, y)
    cramers = [sqrt(i / x.shape[1]) for i in x2]
    return zip(x.columns, x2, p_vals, cramers)


def print_chi2(results):
    pt = PrettyTable()
    pt.field_names = ["feature", "x2", "p-val", "cramers"]
    results = sorted(results, key=lambda x: x[2])
    for row in results:
        pt.add_row(row)
    print(pt)


def run_and_print_chi2(x, y):
    results = run_chi2(x, y)
    print_chi2(results)
    return results
