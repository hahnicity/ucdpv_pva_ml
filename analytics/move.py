from argparse import ArgumentParser
from datetime import datetime
from glob import glob
import os
import re
from shutil import copyfile


def get_corresponding_file(gld_file, raw_files_dir):
    patient_num = gld_file.split("_")[0]
    match = re.search(r"_(\d{2}_\d{2}_\d{2})", gld_file)
    gld_ts = datetime.strptime(match.groups()[0], "%H_%M_%S")
    breath_range = gld_file.split("_")[5]
    patient_files = glob(os.path.join(raw_files_dir, "{}*".format(patient_num), "*.csv"))
    if not patient_files:
        patient_files = glob(os.path.join(raw_files_dir, "{}*".format(patient_num), "*", "*.csv"))
    for f in patient_files:
        basename = os.path.basename(f)
        try:  #old style first
            dt_str = basename.split("_")[-1].split(".")[0]
            dt = datetime.strptime(dt_str, "%H:%M:%S")
        except ValueError:
            try:
                dt_str = "-".join(basename.split("-")[4:7]).split(".")[0]
                dt = datetime.strptime(dt_str, "%H-%M-%S")
            except ValueError:  # new style
                continue
        if dt == gld_ts:
            return f
    else:
        raise Exception("could not find raw file for {}".format(gld_file))


def main():
    parser = ArgumentParser()
    parser.add_argument("raw_files_dir")
    args = parser.parse_args()
    cohorts = ["validation", "derivation"]
    all_gld_files = []
    file_dir = os.path.dirname(__file__)
    for c in cohorts:
        cohort_files = glob(os.path.join(file_dir, c, "files", "*.csv"))

        for gld_file in cohort_files:
            basename = os.path.basename(gld_file)
            raw_file = get_corresponding_file(basename, args.raw_files_dir)
            patient_dir = raw_file.split("/")[-2]
            final_dir = os.path.join(file_dir, c, patient_dir)
            try:
                os.mkdir(final_dir)
            except OSError:
                pass
            copyfile(raw_file, os.path.join(final_dir, os.path.basename(raw_file)))


if __name__ == "__main__":
    main()
