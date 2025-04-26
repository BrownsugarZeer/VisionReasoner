#!/bin/bash

#REASONING_MODEL_PATH="/gpfs/yuqiliu/vision_zero_workdir/20_run_qwen2_5_7b_multiobject_rewardv4_refcoco_lisaplus_n8_nonrepeat/global_step_720/actor/huggingface"
#REASONING_MODEL_PATH="/gpfs/yuqiliu/vision_zero_workdir/21_run_qwen2_5_7b_multiobject_rewardv4_refcoco_lvis_lisaplus_n8_nonrepeat/global_step_200/actor/huggingface"
REASONING_MODEL_PATH="/gpfs/yuqiliu/vision_zero_workdir/22_run_qwen2_5_7b_multiobject_rewardv4_refcoco2k_lvis_lisaplus_grefcoco_n8_nonrepeat/global_step_200/actor/huggingface"

SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*vision_zero_workdir\/(.*)\/actor\/.*/\1/')
TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/ReasonSeg_test"
#TEST_DATA_PATH="/gpfs/yuqiliu/data/Ricky06662/ReasonSeg_val"


TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

NUM_PARTS=8
# Create output directory
mkdir -p $OUTPUT_PATH

# Run 8 processes in parallel
for idx in {0..7}; do
    export CUDA_VISIBLE_DEVICES=$idx
    python evaluation_scripts/evaluation_vision_manus.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 100 &
done

# Wait for all processes to complete
wait

python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH