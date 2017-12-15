#!/usr/local/bin/bash

export fa_patients=(0021RPI0420150513 0126RPI1120160124 0129RPI1620160126)
export derivation_patients=(0001RPI0120150218 0002RPI0320150218 0004RPI0320150223 0007RPI0320150227 0008RPI0120150227 0009RPI0220150302 0010RPI0320150303 0011RPI0320150309 0012RPI0520150316 0013RPI0120150321 0086RPI1920151120 0099RPI0120151219 0122RPI1320160120 0126RPI1120160124 0140RPI1420160208)

export async_type=$1
export classifier=$2
export run_me=$3
export usage="$0 <async type> <classifier> <--run | or not> <additional args>"

if [[ -z $1 ]]; then
    echo "Enter in the type of asynchrony!"
    echo "$usage"
    exit 1
fi

if [[ $async_type = 'fa' || $async_type = 'FA' ]]; then
    export patients=${fa_patients[*]}
elif [[ $async_type = 'dbl' || $async_type = 'BSA' || $async_type = 'DBLA' || $async_type = 'DTA' || $async_type = 'bs' || $async_type = 'co' || $async_type = 'su' || $async_type = 'km' || $async_type = 'dbs' || $async_type = 'DTA+BSA' || $async_type = 'MULTI' ]]; then
    export patients=${derivation_patients[*]}
else
    echo "enter a valid asynchrony type!"
    echo "$usage"
    exit 1
fi

if [[ -z $classifier ]]; then
    echo "input a classifier to use!"
    echo "$usage"
    exit 1
fi

if [[ $async_type = 'fa' || $async_type = 'FA' ]]; then
    export plot_type="per_patient"
elif [[ $classifier = 'km' || $classifier = 'dbs' ]]; then
    export plot_type="clustering"
else
    export plot_type="aggregate"
fi
