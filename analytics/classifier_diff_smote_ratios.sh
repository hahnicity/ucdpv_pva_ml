async_type=$1
run_me=$2
file=$3
feature_set=$4
#classifiers=("cust_vote")
#classifiers=("gbrt" "rf" "etc" "nn")
#classifiers=("gbrt" "etc" "nn" "soft_vote")
#classifiers=("gbrt" "etc" "nn" "rf")
#classifiers=("soft_vote")
#classifiers=("vote")
#classifiers=("etc")
#classifiers=("gbrt")
n_folds=35
classifier="gbrt"

exec() {
    local pickle_file=$1
    local smote_ratio=$2
    # robust classifier works best on tree models, min_max on NN types.
    python learn.py --scaler min_max -c ${classifier} -o results-${classifier}-`basename ${pickle_file}`-smote-ratio-analysis-${feature_set}-${n_folds}-fdb1-${smote_ratio}.csv --write-results -p $pickle_file --winsorize .01 --fdb1  cross_patient_kfold --folds $n_folds --with-smote --smote-ratio $smote_ratio
    if [[ $? -eq 1 ]]; then
        return 1
    fi
}


if [[ $run_me = '--run' ]]; then
    for smote_ratio in $(seq 0.1 0.1 1.0)
    do
        echo "execute on ratio $smote_ratio"
        {
            exec $file $smote_ratio >& /dev/null
        } || {
            continue
        }
    done
fi
if [[ $async_type = 'BSA' || $async_type = 'DBLA' || $async_type = 'DTA' ]]; then
    binary_pva="--binary-pva"
    binary_pva_type=$async_type
else
    binary_pva=
    binary_pva_type=
fi
files=$(find -name "results-${classifier}-`basename ${file}`-smote-ratio-analysis-${feature_set}-${n_folds}-fdb1-*.csv")
find -name "results-${classifier}-`basename ${file}`-smote-ratio-analysis-${feature_set}-${n_folds}-fdb1-*.csv" -exec python calc_averages.py {} \; > tables/${async_type}-${feature_set}-`basename ${file}`-smote-ratio-analysis-fdb1-$classifier.txt
