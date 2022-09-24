#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.05 0.1 0.3 0.5 0.7 #0.9 0.01
do
	for seed in 1 2 3 4 5
	do
		#Baseline
		output_folder=out/train_bs_${job_id}.out
		baseline_args="--sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --seed=$seed --num-attributes 4 --num-values 4 --vocab-size 100 --max-len 10 --check_val_every_n_epoch 500"
		sbatch --job-name "bs_$job_id" --ntasks 2 --time 24:00:00 --output ${output_folder} run.sh $baseline_args
	    	# Feedback
		output_folder=out/train_fb_${job_id}.out
		fb_args=$baseline_args" --noise=0.1 --feedback --length-cost 0.001" #TODO length cost?!
		sbatch --job-name "fb_$job_id" --ntasks 2 --time 24:00:00 --output ${output_folder} run.sh $fb_args
		job_id=$((job_id+1))
	done
done

