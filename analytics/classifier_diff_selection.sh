NUMBER=$3
FEATURES=("tve:tvi-ratio" "tve:tvi-ratio-prev" "eTime-prev" "inst_RR" "epAUC-prev" "epAUC" "tve-prev" "tve:tvi-ratio-prev-prev" "eTime" "I:E-ratio" "ipAUC" "ipAUC-prev" "inst_RR-prev" "I:E-ratio-prev" "iTime" "tve" "I:E-ratio-prev" "I:E-ratio-prev-prev" "eTime-prev-prev" "inst_RR-prev-prev" "ipAUC-prev-prev" "epAUC-prev-prev" "tve-prev-prev" "iTime-prev")

exec() {
    local classifier=$1
    local pickle_file=$2
    TO_USE=("${FEATURES[@]:0:${NUMBER}}")
    python learn.py --winsorize 0.05 --scaler robust --selected-features ${TO_USE[@]} -c $classifier -o results-${classifier}-`basename ${pickle_file}`-greg-${NUMBER}.csv --write-results -p $pickle_file  --n-estimators 50 cross_patient_kfold --folds 15 --with-smote
    if [[ $? -eq 1 ]]; then
        return 1
    fi
}

run_me=$1
file=$2
classifiers=("elm" "gbrt" "ovr_rf" "etc" "nn")

if [[ $run_me = '--run' ]]; then
    for classifier in ${classifiers[*]}
    do
        echo "execute on classifier $classifier"
        {
            exec $classifier $file >& greg${NUMBER}.out
        } || {
            continue
        }
    done
fi
find -name "results-*-`basename ${file}`*-greg-${NUMBER}.csv" -exec python calc_averages.py {} \;
