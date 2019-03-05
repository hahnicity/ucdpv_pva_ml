#!/usr/local/bin/bash
# the list of all experiments to run for the paper
#

## create datasets
mkdir df/
python learn.py --feature-type metadata --async-type bs -npf bs-metadata simple
python learn.py --feature-type metadata --async-type dbl -npf dbl-metadata simple

python learn.py --feature-type bs_chi2 --async-type bs -npf bs-chi2
python learn.py --feature-type dbl_chi2 --async-type dbl -npf dbl-chi2

python learn.py --feature-type bs_curated --async-type bs -npf bs-curated
python learn.py --feature-type dbl_curated --async-type dbl -npf dbl-curated

python learn.py --feature-type dbl_retro --async-type dbl -npf dbl-retro simple
python learn.py --feature-type dbl_retro --async-type bs -npf bs-retro simple

python learn.py --feature-type dbl_retro_chi2 --async-type dbl -npf dbl-retrochi2 simple
python learn.py --feature-type dbl_retro_chi2 --async-type bs -npf bs-retrochi2 simple

python learn.py --feature-type retro_stripped_low-prec  --async-type dbl -npf dbl-retrolowprec simple

python learn.py --feature-type retro_stripped_expert_plus_chi2_5  --async-type bs_dbl_retro_fdb_deriv_val -npf bsdblfdb-expert-chi2-5 simple

## cross patient

# chi2 analysis
./chi2_cross_patient_optimum_analysis.sh BSA rf --run df/bs-metadata.pickle Metadata 16
./chi2_cross_patient_optimum_analysis.sh DTA rf --run df/dbl-metadata.pickle Metadata 16

# cross patient non-retro binary
./classifier_diff.sh BSA --run df/bs-chi2.pickle Metadata
./classifier_diff.sh BSA --run df/bs-curated.pickle Expert
./classifier_diff.sh DTA --run df/dbl-chi2.pickle Metadata
./classifier_diff.sh DTA --run df/dbl-curated.pickle Expert

# cross patient retrospective chi2 optimum
./chi2_cross_patient_optimum_analysis.sh DTA rf --run df/dbl-retro.pickle "Retrospective" 32
./chi2_cross_patient_optimum_analysis.sh BSA rf --run df/bs-retro.pickle "Retrospective" 32

# use chi2 retrospective features to determine what we can use computationally
./classifier_diff.sh DTA --run df/dbl-retrochi2.pickle "Retrospective_Chi-square" fdb1
./classifier_diff.sh BSA --run df/bs-retrochi2.pickle "Retrospective_Chi-square"

# use expert features for dbla
./classifier_diff.sh DTA --run df/dbl-retrolowprec.pickle "Retrospective_Expert" fdb1

# experiments for the multi class classifier
./classifier_diff_undersample.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5.pickle Expert+Chi-square fdb1
./classifier_diff_no_smote.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5.pickle Expert+Chi-square fdb1
./classifier_diff.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5.pickle Expert+Chi-square fdb1
./classifier_diff_smote_ratios.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5.pickle Expert+Chi-square
