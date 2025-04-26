#!/bin/bash

#REASONING_MODEL_PATH="/gpfs/yuqiliu/vision_zero_workdir/15_run_qwen2_5_7b_multiobject_rewardv4_refcoco_lvis_lisaplus_grefcoco/global_step_600/actor/huggingface"
REASONING_MODEL_PATH="/gpfs/yuqiliu/vision_zero_workdir/17_run_qwen2_5_7b_multiobject_rewardv4_refcoco_lvis_lisaplus_grefcoco_n8_nonrepeat/global_step_918/actor/huggingface"
#REASONING_MODEL_PATH="/gpfs/yuqiliu/vision_zero_workdir/16_run_qwen2_5_7b_multiobject_rewardv4_refcoco_lvis_lisaplus_grefcoco_n16/global_step_918/actor/huggingface"

SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*vision_zero_workdir\/(.*)\/actor\/.*/\1/')
#TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/refcocog_test"
TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/refcocog_val"
# TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/refcocoplus_testA"
# TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/refcoco_testA"

TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 通过CUDA_VISIBLE_DEVICES自动计算GPU数量
NUM_PARTS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# 获取最小的GPU编号
MIN_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | sort -n | head -n1)

# Create output directory
mkdir -p $OUTPUT_PATH

# Run processes in parallel
for idx in $(seq 0 $((NUM_PARTS - 1))); do
    export CUDA_VISIBLE_DEVICES=$((idx + MIN_GPU))
    python evaluation_scripts/evaluation_vision_manus.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 250 &
done

# Wait for all processes to complete
wait

python evaluation_scripts/calculate_iou_with_bbox.py --output_dir $OUTPUT_PATH