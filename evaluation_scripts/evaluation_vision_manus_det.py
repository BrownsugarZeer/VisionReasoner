import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import pdb
import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)
    
    return pred_bboxes, pred_points, think_text

def compute_bbox_iou(bboxes1, bboxes2):
    """
    计算两组边界框之间的IOU矩阵
    bboxes1: shape (N, 4) 预测框
    bboxes2: shape (M, 4) 真实框
    返回: shape (N, M) 的IOU矩阵
    """
    # 扩展维度以支持广播
    bboxes1 = np.array(bboxes1)[:, None, :]  # (N, 1, 4)
    bboxes2 = np.array(bboxes2)[None, :, :]  # (1, M, 4)
    
    # 计算交集区域
    x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    # 计算交集面积
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算两个bbox的面积
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 避免除以0
    iou = np.where(union > 0, intersection / union, 0)
    
    return iou

def main():
    args = parse_args()
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    resize_size = 840
    dataset = load_from_disk(args.test_data_path)['test']
    
    # dataset = dataset.select(range(0,1000))
    
    # dataset = load_dataset(args.test_data_path, split='test')
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    # pdb.set_trace()
    dataset = dataset.select(range(start_idx, end_idx))
    
    if 'bbox' in dataset[0]:
        has_bbox = True
    else:
        has_bbox = False
    
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bboxs and points." \
        "Compare the difference between object(s) and find the most closely matched object(s)." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"
    
    messages = []
    id_list = []
    for item in dataset:
        image = item["image"].convert("RGB")
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=item["text"].lower().strip(".\"?!"),
                        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                    )    
                }
            ]
        }]
        messages.append(message)
        id_list.append({
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "cat_id": item['cat_id'],
            "image": image,
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "bbox": item["bbox"] if has_bbox else None,
        })

    all_outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
        batch_id_list = id_list[i:i + args.batch_size]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        
        # pdb.set_trace()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                try:
                    bboxes, points, think = extract_bbox_points_think(
                                            batch_output_text[id_idx], 
                                            batch_id_list[id_idx]["img_width"]/resize_size, 
                                            batch_id_list[id_idx]["img_height"]/resize_size
                                        )
                except Exception as e:
                    # add penalty in this situation
                    print("Reasoning error: ", e, 
                          "Ann_id: ", batch_id_list[id_idx]["ann_id"], 
                          "Image_id: ", batch_id_list[id_idx]["image_id"],
                          "Cat_id: ", batch_id_list[id_idx]["cat_id"])
                    continue
                
                gt_bboxes = batch_id_list[id_idx]["bbox"]
                if gt_bboxes and len(bboxes) > 0:
                    # 使用向量化计算IOU矩阵
                    cost_matrix = -compute_bbox_iou(bboxes, gt_bboxes)  # 使用负IOU作为成本
                    
                    # 使用匈牙利算法进行匹配
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
                    
                    # 为每个预测框分配分数
                    scores = np.zeros(len(bboxes))
                    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                        scores[pred_idx] = -cost_matrix[pred_idx, gt_idx]  # 转回正的IOU值
                    
                    # 添加结果
                    for pred_idx, pred_bbox in enumerate(bboxes):
                        all_outputs.append({
                            "image_id": int(batch_id_list[id_idx]["image_id"]),
                            "ann_id": int(batch_id_list[id_idx]["ann_id"]),
                            "think": think,
                            "category_id": int(batch_id_list[id_idx]["cat_id"]),
                            "bbox": pred_bbox,
                            "score": float(max(scores[pred_idx],0.0))  # 使用匹配得到的分数
                        })
                else:
                    # 如果没有真实框或预测框，则分数为0
                    for pred_bbox in bboxes:
                        all_outputs.append({
                            "image_id": int(batch_id_list[id_idx]["image_id"]),
                            "ann_id": int(batch_id_list[id_idx]["ann_id"]),
                            "think": think,
                            "category_id": int(batch_id_list[id_idx]["cat_id"]),
                            "bbox": pred_bbox,
                            "score": 0.0
                        })
        print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
        
        # clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    
    # Modify the output file name, add idx
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
