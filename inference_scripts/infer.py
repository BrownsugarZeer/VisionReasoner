import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import json
import pdb

from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt

ROUTER_PROMPT = "Given a user instruction, please classify which type of task it is and output the final answer after \"####\".. The types are: 1) Segmentation/detection, 2) Counting, 3) Editing, 4) Caption/QA. The user instruction is: "


QUESTION_TEMPLATE = \
    "Please find \"{Question}\" with bboxs and points." \
    "Compare the difference between object(s) and find the most closely matched object(s)." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_model_path", type=str, default="/gpfs/tyqu/research/verl/checkpoints/verl_grpo_task_route/qwen25_15b/global_step_90/actor/huggingface")
    parser.add_argument("--cognitive_model_path", type=str, default="/gpfs/yuqiliu/vision_zero_workdir/17_run_qwen2_5_7b_multiobject_rewardv4_refcoco_lvis_lisaplus_grefcoco_n8_nonrepeat/global_step_918/actor/huggingface")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--text", type=str, default="How many airplanes are there in this image?")
    parser.add_argument("--image_path", type=str, default="assets/airplanes.png")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output.png")
    return parser.parse_args()


def extract_task_type(response):
    response = response.lower()
    
    number_pattern = r'(?:type|category|task)?\s*(?:is|:)?\s*([1-4])'
    number_matches = re.findall(number_pattern, response)
    if number_matches:
        number = int(number_matches[-1])  # 取最后一个匹配的数字
        if number == 1:
            return 'seg/det'
        elif number == 2:
            return 'count'
        elif number == 3:
            return 'editing'
        elif number == 4:
            return 'vqa'
        
    matches = []
    if any(word in response for word in ['segmentation', 'detection', 'segment', 'detect', 'grounding']):
        matches.append('seg/det')
    if any(word in response for word in ['counting', 'count', 'number']):
        matches.append('count')
    if any(word in response for word in ['editing', 'edit', 'modify']):
        matches.append('editing')
    if any(word in response for word in ['caption', 'qa', 'question', 'answer', 'describe']):
        matches.append('vqa')
        
    return matches[-1] if matches else None

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


def router_task_classification(router_model, router_tokenizer, user_question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ROUTER_PROMPT + user_question}
    ]
    text = router_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = router_tokenizer([text], return_tensors="pt").to(router_model.device)

    generated_ids = router_model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=router_model.config.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = router_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    task_type = extract_task_type(response)
    
    return task_type
    

def cognitive_model_generation(cognitive_model, cognitive_processor, image, user_question, task_type):
    messages = []
    if task_type == 'vqa':
        message = [{
            "role": "user",
            "content": [
            {
                "type": "image", 
                "image": image
            },
            {   
                "type": "text",
                "text": user_question
            }
        ]
        }]        
    else:
        message = [{
            "role": "user",
            "content": [
            {
                "type": "image", 
                "image": image
            },
            {   
                "type": "text",
                "text": QUESTION_TEMPLATE.format(
                    Question=user_question.lower().strip("."),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )    
            }
        ]
        }]
    messages.append(message)

    # Preparation for inference
    text = [cognitive_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    
    #pdb.set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    #pdb.set_trace()
    inputs = cognitive_processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = cognitive_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = cognitive_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # print(output_text[0])
    return output_text[0]
    


def main():
    args = parse_args()
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    router_model = AutoModelForCausalLM.from_pretrained(
        args.router_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    router_tokenizer = AutoTokenizer.from_pretrained(args.router_model_path)
    
    cognitive_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.cognitive_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
        
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    
    cognitive_model.eval()
    
    # default processer
    cognitive_processor = AutoProcessor.from_pretrained(args.cognitive_model_path, padding_side="left")

    print("User question: ", args.text)
    task_type = router_task_classification(router_model, router_tokenizer, args.text)
    print("Task type: ", task_type)
    
    raw_image = PILImage.open(args.image_path)
    raw_image = raw_image.convert("RGB")
    
    original_width, original_height = raw_image.size
    resize_size = 840
    x_factor, y_factor = original_width/resize_size, original_height/resize_size
    image = raw_image.resize((resize_size, resize_size), PILImage.BILINEAR)
    
    output_text = cognitive_model_generation(cognitive_model, cognitive_processor, image, args.text, task_type)
    
    # pdb.set_trace()
    if task_type == 'vqa':
        print("The answer is: ", output_text)
        return 
    
    bboxes, points, think = extract_bbox_points_think(output_text, x_factor, y_factor)
    
    print("Thinking process: ", think)
    
    if task_type == "count":
        print("Total number of interested objects is: ", len(points))
        return 
    # pdb.set_trace()
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((raw_image.height, raw_image.width), dtype=bool)
        segmentation_model.set_image(raw_image)
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = segmentation_model.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0].astype(bool)
            mask_all = np.logical_or(mask_all, mask)
    
    # 修改为1行3列的子图布局
    plt.figure(figsize=(12, 4))
    
    # 第一个子图：原图
    plt.subplot(1, 3, 1)
    plt.imshow(raw_image)
    plt.title('Original Image')
    
    # 第二个子图：原图+bbox
    plt.subplot(1, 3, 2)
    plt.imshow(raw_image)
    # 绘制所有bbox
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        # 可选：绘制对应的点
        for point in points:
            plt.plot(point[0], point[1], 'go', markersize=8)  # 绿色点
    plt.title('Image with Bounding Boxes')
    
    # 第三个子图：mask叠加
    plt.subplot(1, 3, 3)
    plt.imshow(raw_image, alpha=0.6)
    mask_overlay = np.zeros_like(raw_image)
    mask_overlay[mask_all] = [255, 0, 0]
    plt.imshow(mask_overlay, alpha=0.4)
    plt.title('Image with Predicted Mask')
    
    plt.tight_layout()
    plt.savefig(args.output_path)
    plt.close() 

if __name__ == "__main__":
    main()
