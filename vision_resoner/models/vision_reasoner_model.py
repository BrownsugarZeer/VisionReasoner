# Vision-Manus/model/vision_manus_model.py
import torch
import numpy as np
import re
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image as PILImage

from .base_model import (
    BaseVisionModel,
    DetectionModel,
    SegmentationModel,
    CountingModel,
    QAModel
)
from qwen_vl_utils import process_vision_info
from .task_router import TaskRouter

class VisionReasonerModel(BaseVisionModel, DetectionModel, SegmentationModel, CountingModel, QAModel):
    """
    VisionManus model implementing all task interfaces
    """
    def __init__(self, 
                 reasoning_model_path="Ricky06662/VisionReasoner-7B", 
                 segmentation_model_path="facebook/sam2-hiera-large",
                 task_router_model_path="/gpfs/tyqu/research/verl/checkpoints/verl_grpo_task_route/qwen25_15b/global_step_90/actor/huggingface"):
        """
        Initialize the Vision-Manus model with reasoning and segmentation components
        
        Args:
            reasoning_model_path (str): Path to the reasoning model
            segmentation_model_path (str): Path to the segmentation model
        """
        self.resize_size = 840
        
        # Initialize reasoning model
        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.reasoning_model.eval()
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")
        
        # Initialize segmentation model
        self.segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

        self.task_router = TaskRouter(task_router_model_path)
        
        # Template for detection/segmentation tasks
        self.DETECTION_TEMPLATE = \
            "Please find \"{Question}\" with bboxs and points." \
            "Compare the difference between object(s) and find the most closely matched object(s)." \
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
            "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
            "i.e., <think> thinking process here </think>" \
            "<answer>{Answer}</answer>"
        
        # Template for QA tasks
        self.QA_TEMPLATE = "{Question}"

    
    def extract_bbox_points_think(self, output_text, x_factor, y_factor):
        """
        Extract bounding boxes, points, and thinking process from model output
        
        Args:
            output_text (str): Raw output text from the model
            x_factor (float): Scaling factor for x coordinates
            y_factor (float): Scaling factor for y coordinates
            
        Returns:
            tuple: (pred_bboxes, pred_points, think_text, pred_answer)
        """
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        pred_bboxes = []
        pred_points = []
        pred_answer = None
        think_text = ""
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                pred_answer = data
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
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        if think_match:
            think_text = think_match.group(1)
        
        return pred_bboxes, pred_points, think_text, pred_answer
    
    def extract_qa_answer(self, output_text):
        """
        Extract answer for QA tasks
        
        Args:
            output_text (str): Raw output text from the model
            
        Returns:
            dict: Result dictionary with answer and thinking (if available)
        """
        think_pattern = r'<think>([^<]+)</think>'
        think_match = re.search(think_pattern, output_text)
        thinking = think_match.group(1) if think_match else ""
        
        # Remove thinking tags from output to get cleaner answer
        clean_answer = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        
        return {
            "answer": clean_answer,
            "thinking": thinking,
            "full_response": output_text
        }
    
    def generate_masks(self, image, bboxes, points):
        """
        Generate segmentation masks for given image, bounding boxes and points
        
        Args:
            image (PIL.Image): Input image
            bboxes (list): List of bounding boxes
            points (list): List of points
            
        Returns:
            numpy.ndarray: Combined segmentation mask
        """
        img_height, img_width = image.height, image.width
        mask_all = np.zeros((img_height, img_width), dtype=bool)
        
        if not bboxes or not points:
            return mask_all
        
        try:
            self.segmentation_model.set_image(image)
            
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = self.segmentation_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                mask = masks[0].astype(bool)
                mask_all = np.logical_or(mask_all, mask)
                
            return mask_all
        except Exception as e:
            print(f"Error generating masks: {e}")
            return mask_all
    
    def _generate_model_output(self, images, instructions, template, batch_mode=False):
        """
        Generate raw model output for images and instructions
        
        Args:
            images (PIL.Image or List[PIL.Image]): Input image(s)
            instructions (str or List[str]): Text instruction(s)/query(ies)
            template (str): Template to use for the prompt
            batch_mode (bool): Whether to process in batch mode
            
        Returns:
            tuple: (output_texts, scale_factors)
        """
        if not batch_mode:
            images = [images]
            instructions = [instructions]
        
        batch_messages = []
        scale_factors = []
        
        for image, instruction in zip(images, instructions):
            # Prepare image
            original_width, original_height = image.size
            x_factor, y_factor = original_width/self.resize_size, original_height/self.resize_size
            scale_factors.append((x_factor, y_factor))
            resized_image = image.resize((self.resize_size, self.resize_size), PILImage.BILINEAR)
            
            # Format text based on template
            if "{Question}" in template:
                formatted_text = template.format(
                    Question=instruction.lower().strip(".\"?!"),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )
            else:
                formatted_text = template
                
            # Create message
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": resized_image
                    },
                    {   
                        "type": "text",
                        "text": formatted_text
                    }
                ]
            }]
            batch_messages.append(message)
        
        # Prepare for batch inference
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate output
        generated_ids = self.reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        if not batch_mode:
            return output_texts[0], scale_factors[0]
        return output_texts, scale_factors
    
    # BaseVisionModel implementation
    def process_single_image(self, image, instruction, return_task_type=False):
        """
        Process a single image with given instruction
        
        Args:
            image (PIL.Image): Input image
            instruction (str): Text instruction or query
            
        Returns:
            dict: Results dictionary
        """
        # Determine task type based on instruction
        task_type = self.task_router.route_task(instruction)
        
        if task_type == "segmentation":
            result = self.segment_objects(image, instruction)
        elif task_type == "detection":
            result = self.detect_objects(image, instruction)
        elif task_type == "counting":
            result = self.count_objects(image, instruction)
        else:  # Default to QA
            result = self.answer_question(image, instruction)
        
        if return_task_type:
            return result, task_type
        else:
            return result
    
    def process_batch(self, batch_images, batch_instructions):
        """
        Process a batch of images with given instructions
        
        Args:
            batch_images (list): List of PIL Images
            batch_instructions (list): List of text instructions or queries
            
        Returns:
            list: List of result dictionaries
        """
        results = []
        for image, instruction in zip(batch_images, batch_instructions):
            result = self.process_single_image(image, instruction)
            results.append(result)
        return results
    
    # DetectionModel implementation
    def detect_objects(self, image, query):
        """
        Detect objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to detect
            
        Returns:
            dict: Results with bounding boxes and scores
        """
        try:
            output_text, (x_factor, y_factor) = self._generate_model_output(
                image,
                query,
                self.DETECTION_TEMPLATE
            )
            
            bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                output_text, 
                x_factor, 
                y_factor
            )
            
            # Assign confidence scores (all 1.0 as the model doesn't provide them)
            scores = [1.0] * len(bboxes)
            
            return {
                "bboxes": bboxes,
                "points": points,
                "scores": scores,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            raise
            print(f"Error in detection: {e}")
            return {
                "bboxes": [],
                "points": [],
                "scores": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def detect_objects_batch(self, images, queries):
        """
        Detect objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of detection results
        """
        try:
            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text, (x_factor, y_factor) in zip(output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                scores = [1.0] * len(bboxes)
                results.append({
                    "bboxes": bboxes,
                    "points": points,
                    "scores": scores,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch detection: {e}")
            return [{
                "bboxes": [],
                "points": [],
                "scores": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for _ in range(len(images))]
    
    # SegmentationModel implementation
    def segment_objects(self, image, query):
        """
        Segment objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to segment
            
        Returns:
            dict: Results with masks and bounding boxes
        """
        try:
            output_text, (x_factor, y_factor) = self._generate_model_output(
                image,
                query,
                self.DETECTION_TEMPLATE
            )
            
            bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                output_text, 
                x_factor, 
                y_factor
            )
            
            masks = self.generate_masks(image, bboxes, points)
            
            return {
                "masks": masks,
                "bboxes": bboxes,
                "points": points,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            print(f"Error in segmentation: {e}")
            img_height, img_width = image.height, image.width
            return {
                "masks": np.zeros((img_height, img_width), dtype=bool),
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def segment_objects_batch(self, images, queries):
        """
        Segment objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of segmentation results
        """
        try:
            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for image, output_text, (x_factor, y_factor) in zip(images, output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                masks = self.generate_masks(image, bboxes, points)
                results.append({
                    "masks": masks,
                    "bboxes": bboxes,
                    "points": points,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch segmentation: {e}")
            return [{
                "masks": np.zeros((img.height, img.width), dtype=bool),
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for img in images]
    
    # CountingModel implementation
    def count_objects(self, image, query):
        """
        Count objects in an image based on a query
        
        Args:
            image: Input image
            query: Text query describing what to count
            
        Returns:
            dict: Results with count and bounding boxes
        """
        try:
            output_text, (x_factor, y_factor) = self._generate_model_output(
                image,
                query,
                self.DETECTION_TEMPLATE
            )
            
            bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                output_text, 
                x_factor, 
                y_factor
            )
            
            count = len(bboxes)
            
            return {
                "count": count,
                "bboxes": bboxes,
                "points": points,
                "thinking": thinking,
                "full_response": output_text,
                "pred_answer": pred_answer
            }
        except Exception as e:
            print(f"Error in counting: {e}")
            return {
                "count": 0,
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            }
    
    def count_objects_batch(self, images, queries):
        """
        Count objects in a batch of images
        
        Args:
            images: List of input images
            queries: List of text queries
            
        Returns:
            list: List of counting results
        """
        try:
            output_texts, scale_factors = self._generate_model_output(
                images,
                queries,
                self.DETECTION_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text, (x_factor, y_factor) in zip(output_texts, scale_factors):
                bboxes, points, thinking, pred_answer = self.extract_bbox_points_think(
                    output_text, 
                    x_factor, 
                    y_factor
                )
                
                count = len(bboxes)
                results.append({
                    "count": count,
                    "bboxes": bboxes,
                    "points": points,
                    "thinking": thinking,
                    "full_response": output_text,
                    "pred_answer": pred_answer
                })
            return results
        except Exception as e:
            print(f"Error in batch counting: {e}")
            return [{
                "count": 0,
                "bboxes": [],
                "points": [],
                "thinking": "",
                "full_response": "",
                "pred_answer": None
            } for _ in range(len(images))]
    
    # QAModel implementation
    def answer_question(self, image, question):
        """
        Answer a question about an image
        
        Args:
            image: Input image
            question: Text question
            
        Returns:
            dict: Results with answer and thinking (if available)
        """
        try:
            output_text, _ = self._generate_model_output(
                image,
                question,
                self.QA_TEMPLATE
            )
            
            result = self.extract_qa_answer(output_text)
            return result
        except Exception as e:
            print(f"Error in QA: {e}")
            return {
                "answer": "",
                "thinking": "",
                "full_response": ""
            }
    
    def answer_questions_batch(self, images, questions):
        """
        Answer questions about a batch of images
        
        Args:
            images: List of input images
            questions: List of text questions
            
        Returns:
            list: List of QA results
        """
        try:
            output_texts, _ = self._generate_model_output(
                images,
                questions,
                self.QA_TEMPLATE,
                batch_mode=True
            )
            
            results = []
            for output_text in output_texts:
                result = self.extract_qa_answer(output_text)
                results.append(result)
            return results
        except Exception as e:
            print(f"Error in batch QA: {e}")
            return [{
                "answer": "",
                "thinking": "",
                "full_response": ""
            } for _ in range(len(images))]