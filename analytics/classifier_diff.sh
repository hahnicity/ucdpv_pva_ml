async_type=$1
run_me=$2
file=$3
feature_set=$4
#classifiers=("cust_vote")
#classifiers=("gbrt" "rf" "etc" "nn")
classifiers=("gbrt" "etc" "nn" "soft_vote")
#classifiers=("gbrt" "etc" "nn" "rf")
#classifiers=("soft_vote")
#classifiers=("vote")
#classifiers=("etc")
#classifiers=("gbrt")
n_folds=35
fdb_type=$5
fdb_val=$6

exec() {
    local classifier=$1
    local pickle_file=$2
    # robust classifier works best on tree models, min_max on NN types.
    if [[ $classifier = 'gbrt' || $classifier = 'ovr_rf' || $classifier = 'etc' || $classifier = 'rf' ]]; then
        scaler=robust
    else
        scaler=min_max
    fi
    if [[ -z $fdb_type ]]; then
        local fdb=
    else
        local fdb=--${fdb_type}
    fi
    echo "python learn.py --scaler $scaler -c $classifier -o results-${classifier}-`basename ${pickle_file}`-classifier-analysis-${feature_set}-${n_folds}-${fdb_type}.csv --write-results -p $pickle_file  --n-estimators 50 --winsorize .01 ${fdb} ${fdb_val} cross_patient_kfold --folds $n_folds --with-smote"
    python learn.py --scaler $scaler -c $classifier -o results-${classifier}-`basename ${pickle_file}`-classifier-analysis-${feature_set}-${n_folds}-${fdb_type}.csv --write-results -p $pickle_file  --n-estimators 50 --winsorize .01 ${fdb} ${fdb_val} cross_patient_kfold --folds $n_folds --with-smote
    if [[ $? -eq 1 ]]; then
        return 1
    fi
}


if [[ $run_me = '--run' ]]; then
    for classifier in ${classifiers[*]}
    do
        echo "execute on classifier $classifier"
        {
            exec $classifier $file #>& /dev/null
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
files=$(find -name "results-*`basename ${file}`-classifier-analysis-${feature_set}-${n_folds}-${fdb_type}.csv")
#files=$(find -name "results-*`basename ${file}`-classifier-analysis-${n_folds}.csv")
find -name "results-*`basename ${file}`-classifier-analysis-${feature_set}-${n_folds}-${fdb_type}.csv" -exec python calc_averages.py {} \; > tables/${async_type}-${feature_set}-`basename ${file}`-cross-patient-analysis-${fdb_type}.txt
python plot_auc.py --x-type bin --x-lab classifier --title "${feature_set}" --png "images/${async_type}-${feature_set}-`basename ${file}`-cross-patient-with-voting-${n_folds}-${fdb_type}.png" classifier $files $binary_pva $binary_pva_type
