#!/bin/bash
set -e

MODEL_TYPE="vision_reasoner"  # Model type: qwen or vision_reasoner
TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/coco_val"

# Extract model name and test dataset name for output directory
TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./detection_eval_results/${MODEL_TYPE}/${TEST_NAME}"

# Customize GPU array here - specify which GPUs to use
GPU_ARRAY=(0 1 2 3 4 5 6 7)  # Example: using GPUs 0, 1, 2, 3
NUM_PARTS=${#GPU_ARRAY[@]}

# Create output directory
mkdir -p $OUTPUT_PATH

# Function to handle errors and cleanup
trap 'echo "Error: A process has failed, terminating all processes"; kill $(jobs -p) 2>/dev/null; exit 1' ERR

# Run processes in parallel
for i in $(seq 0 $((NUM_PARTS-1))); do
    gpu_id=${GPU_ARRAY[$i]}
    process_idx=$i  # 0-based indexing for process
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    (
        python evaluation/evaluation_coco.py \
            --model $MODEL_TYPE \
            --output_path $OUTPUT_PATH \
            --test_data_path $TEST_DATA_PATH \
            --idx $process_idx \
            --num_parts $NUM_PARTS \
            --batch_size 16 || { echo "1" > /tmp/process_status.$$; kill -TERM -$$; }
    ) &
done

# Wait for all processes to complete
wait

COCO_GT_JSON_PATH="/gpfs/yuqiliu/data/Segmentation/coco/instances_val2017.json"

# Calculate COCO AP metrics
python evaluation/calculate_coco_ap.py --output_dir $OUTPUT_PATH --gt_json_path "/gpfs/yuqiliu/data/Segmentation/coco/instances_val2017.json"
#python evaluation_scripts/calculate_coco_ap.py --output_dir "/gpfs/tyqu/research/vision-manus/Vision-Manus/detection_eval_results/qwen/coco_val" --gt_json_path "/gpfs/yuqiliu/data/Segmentation/coco/instances_val2017.json"
#python evaluation_scripts/calculate_coco_ap.py --output_dir "/gpfs/tyqu/research/vision-manus/Vision-Manus/detection_eval_results/qwen2/coco_val" --gt_json_path "/gpfs/yuqiliu/data/Segmentation/coco/instances_val2017.json"
#python evaluation_scripts/calculate_coco_ap.py --output_dir $OUTPUT_PATH --gt_json_path $COCO_GT_JSON_PATH