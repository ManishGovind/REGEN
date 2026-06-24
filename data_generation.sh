#!/bin/bash

'''
This script is used to generate data for the new task.

Usage:
./data_generation.sh <task_suite_name> <method_name> <output_runid_file> <dataset_stats_path>

Example:
bash data_generation.sh libero_goal cosmos_predict2_2b_480p_libero_goal_base_stage eval-on-tasks0-5 cl_base_stage_libero_goal_success_only
'''
task_suite_name=$1
method_name=$2
task_ids_to_run=$3
dataset_stats_path=$4




uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero_cl_stage_inference_only \
    --ckpt_path /path/to/base_stage_policy/checkpoints/model.pt \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path $dataset_stats_path \
    --t5_text_embeddings_path /path/to/t5_embeddings.pkl \
    --trained_with_image_aug False \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name $task_suite_name \
    --local_log_dir ./logs/$task_suite_name \
    --randomize_seed False \
    --data_collection True \
    --num_trials_per_task 10 \
    --available_gpus "1" \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note "REGEN-DataGeneration-${output_runid_file}-${method_name}-seed195" \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images False \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --data_generation True \
    --task_ids_to_run "$task_ids_to_run" \
    --eval_hdf5_path /path/to/eval_hdf5_path.hdf5 \
    --save_rollout_video True 
done
