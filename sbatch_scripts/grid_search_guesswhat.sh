#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.01 #0.001 0.01 0.1 #0.01 0.05 0.2 #0.001 #0.1 # 0.01 0.3 #0.5 # 0.1 0.3 0.5 0.7 #0.9 0.01 0.05
do
	for seed in 2 3 #4 5 #6 7 8 9 10
	do
		num_senders=1
		num_receivers=1
		num_attributes=4
		num_values=4
		vocab_size=20
		max_len=2
		patience=100 
		noise=0.0 
		length_cost=0.1
		training_time="00:08:00:00"
		job_time="10:00:00"
		vocab_size_feedback=2 #100 #TODO

		baseline_args="-A eqb@a100 -C a100 --nodes=1 --nodes=1 --gres=gpu:1 --cpus-per-task=10 --hint=nomultithread --baseline-type 'mean' --precision=16 --accelerator 'gpu' --sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --seed=$seed --num-senders=$num_senders --num-receivers=$num_receivers --patience=$patience --vocab-size=$vocab_size --max-len=$max_len --check_val_every_n_epoch 10 --num-workers 4 --max_time=$training_time --sender-layer-norm --receiver-layer-norm --discrimination-num-objects 2 --sender-embed-dim 10 --receiver-embed-dim 10 --length-cost $length_cost --noise=$noise --guesswhat"

		#Baseline
		output_folder=out_gpu/train_bs_${job_id}.out
		sbatch --job-name "bs_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition --gres gpu:1 run_jean_zay.sh $baseline_args

		# Feedback
                output_folder=out_gpu/train_fb_${job_id}.out
                fb_args="$baseline_args --feedback --vocab-size-feedback=$vocab_size_feedback"
                sbatch --job-name "fb_$job_id" --ntasks 2 --time=$job_time --output ${output_folder} --partition=$partition --gres gpu:1 run_jean_zay.sh $fb_args


		job_id=$((job_id+1))
	done
done

