#!/bin/sh


for i in {1..7}
do
    echo "Running job $i"
    bsub < job_scripts/job_arNet.sh
done