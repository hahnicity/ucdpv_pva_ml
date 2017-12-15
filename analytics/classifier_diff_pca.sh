PCA=$3

exec() {
    local classifier=$1
    local pickle_file=$2
    python learn.py -c $classifier -o results-${classifier}-`basename ${pickle_file}`-pca-${PCA}.csv --write-results -p $pickle_file  --n-estimators 50 --pca ${PCA} --scaler robust cross_patient_kfold --folds 15 --with-smote
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
            exec $classifier $file >& pca${PCA}.out
        } || {
            continue
        }
    done
fi
find -name "results-*-`basename ${file}`*-pca-${PCA}.csv" -exec python calc_averages.py {} \;
