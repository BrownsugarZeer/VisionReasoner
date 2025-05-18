import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class TaskRouter:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """初始化任务路由器"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.prompt_template = "Given a user instruction, please classify which type of task it is and output the final answer after \"####\".. The types are: 1) Segmentation/detection, 2) Counting, 3) Editing, 4) Caption/QA. The user instruction is: "
        
        self.labels = ['seg/det', 'count', 'editing', 'vqa']

    def route_task(self, instruction):
        """将输入指令路由到对应的任务类别
        
        Args:
            instruction: 用户输入的指令
            
        Returns:
            dict: 包含预测类别和置信度的字典
        """
        # 获取模型响应
        response = self._get_model_response(instruction)
        
        # 提取类别
        predicted_category = self._extract_category(response)
        
        return predicted_category
    
    def route_batch(self, instructions):
        """批量路由任务
        
        Args:
            instructions: 指令列表
            
        Returns:
            list: 结果字典列表
        """
        # 获取批量响应
        responses = self._get_model_responses(instructions)
        
        results = []
        for instruction, response in zip(instructions, responses):
            category = self._extract_category(response)
            results.append(category)
            
        return results
    
    def _get_model_response(self, instruction):
        """获取单个指令的模型响应"""
        return self._get_model_responses([instruction])[0]
    
    def _get_model_responses(self, instructions):
        """批量获取模型响应
        
        Args:
            instructions: 指令列表
            
        Returns:
            list: 响应列表
        """
        # 构建批量消息
        message_batch = [
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": self.prompt_template + instruction}
            ]
            for instruction in instructions
        ]
        
        # 批量处理模板
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 批量tokenize
        model_inputs = self.tokenizer(
            text_batch, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            # 只保留新生成的token
            generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return responses
    
    def _extract_category(self, response):
        """从模型响应中提取预测的类别"""
        response = response.lower()
        
        # 1. 查找所有数字匹配
        number_pattern = r'(?:type|category|task)?\s*(?:is|:)?\s*([1-4])'
        number_matches = re.findall(number_pattern, response)
        if number_matches:
            number = int(number_matches[-1])  # 取最后一个匹配的数字
            if number == 1:
                return 'segmentation'
            elif number == 2:
                return 'counting'
            elif number == 3:
                return 'editing'
            elif number == 4:
                return 'vqa'
            
        # 2. 关键词匹配 - 查找所有匹配的关键词
        matches = []
        if any(word in response for word in ['segmentation', 'detection', 'segment', 'detect', 'grounding']):
            matches.append('segmentation')
        if any(word in response for word in ['counting', 'count', 'number']):
            matches.append('counting')
        if any(word in response for word in ['editing', 'edit', 'modify']):
            matches.append('editing')
        if any(word in response for word in ['caption', 'qa', 'question', 'answer', 'describe']):
            matches.append('vqa')
            
        # 返回最后一个匹配的类别
        return matches[-1] if matches else "vqa"