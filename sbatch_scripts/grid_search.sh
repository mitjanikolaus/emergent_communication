#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.1 0.3 0.5 0.7 #0.9 0.01 0.05 # TODO
do
	for seed in 1 2 3 4 5 # 6 7 8 9 10
	do
		#Baseline
		output_folder=out/train_bs_${job_id}.out
		baseline_args="--sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --seed=$seed --num-attributes 4 --num-values 4 --vocab-size 100 --max-len 10 --check_val_every_n_epoch 500 --num-workers 2"
		#sbatch --job-name "bs_$job_id" --ntasks 2 --time 36:00:00 --output ${output_folder} run.sh $baseline_args
	    	
		# Baseline + noise
		output_folder=out/train_noise_${job_id}.out
		noise_args="$baseline_args --noise 0.1"
		#sbatch --job-name "n_$job_id" --ntasks 2 --time 36:00:00 --output ${output_folder} run.sh $noise_args

		# Baseline + noise + length cost
		output_folder=out/train_noise_length_cost_${job_id}.out
                noise_length_cost_args="$baseline_args --noise 0.1 --length-cost 0.001"
                #sbatch --job-name "n_l_$job_id" --ntasks 2 --time 36:00:00 --output ${output_folder} run.sh $noise_length_cost_args

		# Feedback
		output_folder=out/train_fb_${job_id}.out
		fb_args="$baseline_args --noise=0.1 --feedback --length-cost 0.001" #TODO length cost?!
		sbatch --job-name "fb_$job_id" --ntasks 2 --time 36:00:00 --output ${output_folder} run.sh $fb_args

		job_id=$((job_id+1))
	done
done

