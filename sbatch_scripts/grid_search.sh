#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.1 0.3 0.5 0.7
do
	#Baseline
	output_folder=out/train_baseline_${job_id}.out
	baseline_args="--sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --num-features 4 --num-values 4 --vocab-size 100 --max-len 10"
	#--log-topsim-on-validation --log-posdis-on-validation --log-bosdis-on-validation
	sbatch --job-name "train_baseline" --ntasks 8 --time 8:00:00 --output ${output_folder} run.sh $baseline_args
    	# CRs
	output_folder=out/train_cr_${job_id}.out
	cr_args=$baseline_args" --noise=0.1 --clarification-requests"
	sbatch --job-name "train_cr" --ntasks 8 --time 8:00:00 --output ${output_folder} run.sh $cr_args
	job_id=$((job_id+1))
done

