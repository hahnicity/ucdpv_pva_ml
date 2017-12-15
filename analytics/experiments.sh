#!/usr/local/bin/bash
# the list of all experiments to run for the paper
#

## cross patient

# chi2 analysis
./chi2_cross_patient_optimum_analysis.sh BSA rf --run df/bs-metadata-2017-04-27.pickle Metadata 16
./chi2_cross_patient_optimum_analysis.sh DTA rf --run df/dbl-metadata-2017-04-28.pickle Metadata 16


# cross patient non-retro binary
./classifier_diff.sh BSA --run df/bs-chi2-2017-05-29.pickle Metadata
./classifier_diff.sh BSA --run df/bs-curated-2017-04-28.pickle Expert
./classifier_diff.sh DTA --run df/dbl-chi2-2017-05-29.pickle Metadata
./classifier_diff.sh DTA --run df/dbl-curated-2017-04-28.pickle Expert

# cross patient retrospective chi2 optimum
./chi2_cross_patient_optimum_analysis.sh DTA rf --run df/dbl-retro-2017-04-28.pickle "Retrospective" 32
./chi2_cross_patient_optimum_analysis.sh BSA rf --run df/bs-retro-2017-04-28.pickle "Retrospective" 32

# use chi2 retrospective features to determine what we can use computationally
./classifier_diff.sh DTA --run df/dbl-retrochi2-2017-05-29.pickle "Retrospective Chi-square" fdb1
./classifier_diff.sh BSA --run df/bs-retro-2017-04-28.pickle "Retrospective Chi-square"

# use expert features for dbla
./classifier_diff.sh DTA --run df/dbl-retrolowprec-2017-04-28.pickle "Retrospective Expert" fdb1

# experiments for the multi class classifier
./classifier_diff_undersample.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5-2017-06-26.pickle Expert+Chi-square fdb1
./classifier_diff_no_smote.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5-2017-06-26.pickle Expert+Chi-square fdb1
./classifier_diff.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5-2017-06-26.pickle Expert+Chi-square fdb1
./classifier_diff_smote_ratios.sh "DTA+BSA" --run df/bsdblfdb-expert-chi2-5-2017-06-26.pickle Expert+Chi-square
