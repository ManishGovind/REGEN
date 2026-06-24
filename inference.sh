#!/bin/bash
# Usage: ./inference.sh <task_suite_name> <method_name> <output_runid_file> <dataset_stats_path>
#
# Example:
#   ./inference.sh libero_goal my_run_id /path/to/dataset_statistics.json

set -euo pipefail

task_suite_name=$1
method_name=$2
output_runid_file=$3
dataset_stats_path=$4


for iter in $(seq 2000 1000 2000); do
  ckpt_iter=$(printf "%09d" "$iter")

  uv run --extra cu128 --group libero --python 3.10 \
    python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
      --config cosmos_predict2_2b_480p_libero_cl_stage_inference_only \
      --ckpt_path /workspace/checkpoints/imaginaire4-output/cosmos_policy/cosmos_v2_finetune/${method_name}/checkpoints/iter_${ckpt_iter}/model.pt \
      --config_file cosmos_policy/config/config.py \
      --use_wrist_image True \
      --use_proprio True \
      --normalize_proprio True \
      --unnormalize_actions True \
      --dataset_stats_path ${dataset_stats_path} \
      --t5_text_embeddings_path /path/to/t5_embeddings.pkl \
      --trained_with_image_aug True \
      --chunk_size 16 \
      --num_open_loop_steps 16 \
      --task_suite_name $task_suite_name \
      --local_log_dir ./logs/$task_suite_name \
      --randomize_seed False \
      --data_collection False \
      --num_trials_per_task 50 \
      --available_gpus "1" \
      --seed 195 \
      --use_variance_scale False \
      --deterministic True \
      --run_id_note "Eval-${output_runid_file}-${method_name}-chkpt${iter}-seed196" \
      --ar_future_prediction False \
      --ar_value_prediction False \
      --use_jpeg_compression True \
      --flip_images True \
      --num_denoising_steps_action 5 \
      --num_denoising_steps_future_state 1 \
      --num_denoising_steps_value 1 \
      --task_ids_to_run "$task_ids_to_run" \
      --use_wandb False \
      --save_rollout_video True

done
