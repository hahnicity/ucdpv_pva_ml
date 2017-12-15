async_type=$1
run_me=$2
file=$3
feature_set=$4
#classifiers=("cust_vote")
classifiers=("gbrt" "rf" "etc" "soft_vote")
#classifiers=("gbrt" "etc" "nn" "soft_vote")
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
    python learn.py --scaler $scaler -c $classifier -o results-${classifier}-`basename ${pickle_file}`-classifier-analysis-no-smote-${fdb_type}.csv --write-results -p $pickle_file  --n-estimators 50 --winsorize .01 ${fdb} ${fdb_val} cross_patient_kfold --folds 35
    if [[ $? -eq 1 ]]; then
        return 1
    fi
}


if [[ $run_me = '--run' ]]; then
    for classifier in ${classifiers[*]}
    do
        echo "execute on classifier $classifier"
        {
            exec $classifier $file >& /dev/null
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
files=$(find -name "results-*`basename ${file}`-classifier-analysis-no-smote-${fdb_type}.csv")
find -name "results-*`basename ${file}`*-classifier-analysis-no-smote-${fdb_type}.csv" -exec python calc_averages.py {} \; > tables/${async_type}-`basename ${file}`-cross-patient-analysis-no-smote-${fdb_type}.txt
python plot_auc.py --x-type bin --x-lab classifier --title "${async_type} ${feature_set} Cross Patient Non-SMOTE Analysis" --png "images/${async_type}-${feature_set}-cross-patient-no-smote-with-voting-cls-${fdb_type}.png" classifier $files $binary_pva $binary_pva_type
