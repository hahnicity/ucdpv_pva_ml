# Using Machine Learning to Detect Patient Ventilator Asynchrony

## Paper
Accepted for publication to Methods of Information in Medicine.

## Setup

Use anaconda for the primary setup, and then setuptools for the rest

    # make sure to install your matplotlib drivers. I use pyqt, but maybe you use something else.
    conda install pandas scipy numpy matplotlib scikit-learn pyqt
    pip install -e .

## Dataset
If you are interested in reproducing our results please contact lab PI Jason Adams
for access to our dataset. Contact `jyadams@ucdavis.edu`.

Once you have accessed the dataset you can setup the data into the analytics directory

    cd analytics
    mv path/to/dataset/cohort_derivation .
    mv path/to/dataset/cohort_validation .

## Analytics
After obtaining the dataset you can reproduce by:

    cd analytics
    ./experiments.sh
