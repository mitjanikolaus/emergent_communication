#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.01 0.1 0.3 #0.5 # 0.1 0.3 0.5 0.7 #0.9 0.01 0.05
do
	for seed in 2 #3 #4 5 #6 7 8 9 10 #TODO
	do
		num_attributes=4
		num_values=100
		num_senders=1
		num_receivers=1
		vocab_size=20
		max_len=6
		test_set_size=-1
		patience=100 # TODO
		noise=0.0 #TODO
		length_cost=0 #0.001 #TODO
		training_time="00:24:00:00"
		job_time="36:00:00"
		vocab_size_feedback=20 #100 #TODO
		partition=besteffort #gpu

		baseline_args="--sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --seed=$seed --num-attributes=$num_attributes --num-values=$num_values --num-senders=$num_senders --num-receivers=$num_receivers --patience=$patience --vocab-size=$vocab_size --max-len=$max_len --check_val_every_n_epoch 10 --num-workers 2 --max_time=$training_time --sender-layer-norm --receiver-layer-norm --log-topsim-on-validation --test-set-size=$test_set_size --discrimination-game --discrimination-num-objects 5 --hard-distractors" # TODO --stochastic-receiver

		#Baseline
		output_folder=out_gpu/train_bs_${job_id}.out
		sbatch --job-name "bs_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition run.sh $baseline_args # --gres gpu:1 #TODO

                #Reset
                output_folder=out_gpu/train_reset_${job_id}.out
                reset_args="$baseline_args --reset-parameters --reset-parameters-interval=1"
                #sbatch --job-name "reset_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition run.sh $reset_args # --gres gpu:1 #TODO

	    	
		# Baseline + noise
		output_folder=out_gpu/train_noise_${job_id}.out
		noise_args="$baseline_args --noise=$noise"
#		sbatch --job-name "n_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition run.sh $noise_args 

		# Self-repair
		output_folder=out_gpu/train_sr_${job_id}.out
		self_repair_args="$baseline_args --noise=$noise --self-repair"
#		sbatch --job-name "sr_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition run.sh $self_repair_args

		# Feedback
                output_folder=out_gpu/train_fb_${job_id}.out
                fb_args="$baseline_args --noise=$noise --feedback --vocab-size-feedback=$vocab_size_feedback"
               # sbatch --job-name "fb_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition run.sh $fb_args







		# Baseline + noise + length cost
                #output_folder=out/train_noise_length_cost_${job_id}.out
                #noise_length_cost_args="$baseline_args --noise 0.1 --length-cost=$length_cost"
                #sbatch --job-name "n_l_$job_id" --ntasks 2 --time 36:00:00 --output ${output_folder} --partition besteffort run.sh $noise_length_cost_args


		# Feedback with length cost
                #output_folder=out/train_fb_l_${job_id}.out
		#fb_l_args="$baseline_args --noise=0.1 --feedback --length-cost=$length_cost"
                #sbatch --job-name "fb_l_$job_id" --ntasks 2 --time 36:00:00 --output ${output_folder} --partition besteffort run.sh $fb_l_args


		job_id=$((job_id+1))
	done
done

