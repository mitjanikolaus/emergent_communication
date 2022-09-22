#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.01 0.05 0.1 0.3 0.5 0.7 0.9
do
	for seed in 1 2 3
	do
		#Baseline
		output_folder=out/train_baseline_${job_id}.out
		baseline_args="--sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --seed=$seed --num-attributes 4 --num-values 4 --vocab-size 100 --max-len 10 --num-senders 2 --num-receivers 2"
		sbatch --job-name "bs_$job_id" --ntasks 8 --time 24:00:00 --output ${output_folder} run.sh $baseline_args
	    	# CRs
		output_folder=out/train_cr_${job_id}.out
		cr_args=$baseline_args" --noise=0.1 --clarification-requests"
		sbatch --job-name "cr_$job_id" --ntasks 8 --time 24:00:00 --output ${output_folder} run.sh $cr_args
		job_id=$((job_id+1))
	done
done

