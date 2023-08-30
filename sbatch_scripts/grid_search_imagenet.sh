#!/bin/bash

set -e

job_id=0

for entropy_coeff in 0.01 #0.001 0.01 0.1 #0.01 0.05 0.2 #0.001 #0.1 # 0.01 0.3 #0.5 # 0.1 0.3 0.5 0.7 #0.9 0.01 0.05
do
	for seed in 2 #2 3 #4 5 #6 7 8 9 10 #TODO
	do
		num_senders=1
		num_receivers=1
		vocab_size=20
		max_len=10
		patience=100 # TODO
		noise=0.0 #TODO
		length_cost=0 #0.001 #TODO
		training_time="00:20:00:00"
		job_time="22:00:00"
		vocab_size_feedback=20 #100 #TODO

		baseline_args="--baseline-type 'mean' --precision=16 --accelerator 'gpu' --sender-entropy-coeff=$entropy_coeff --receiver-entropy-coeff=$entropy_coeff --seed=$seed --num-senders=$num_senders --num-receivers=$num_receivers --patience=$patience --vocab-size=$vocab_size --max-len=$max_len --val_check_interval 10000 --num-workers 10 --max_time=$training_time --sender-layer-norm --receiver-layer-norm --imagenet --discrimination-num-objects 2 --sender-embed-dim 10 --receiver-embed-dim 10"

		#Baseline
		output_path=out_gpu/train_bs_${job_id}.out
		sbatch --account "eqb@v100" --job-name "bs_$job_id" --time=$job_time --output ${output_path} --error ${output_path} --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=10 --hint=nomultithread run_jean_zay.sh $baseline_args

		# Feedback
                output_folder=out_gpu/train_fb_${job_id}.out
                fb_args="$baseline_args --noise=$noise --feedback --vocab-size-feedback=$vocab_size_feedback"
                #sbatch --account "eqb@v100" --job-name "fb_$job_id" --time=$job_time --output ${output_folder} --error ${output_path} --nodes=1 --ntasks-per-node=1 --gres gpu:1 --cpus-per-task=10 --hint=nomultithread run_jean_zay.sh $fb_args


		job_id=$((job_id+1))
	done
done

