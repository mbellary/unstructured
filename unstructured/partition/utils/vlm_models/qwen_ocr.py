import copy
import numpy as np
import torch
import ast

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords
from unstructured.partition.utils.constants import Source, QWEN_CONF
from unstructured.utils import requires_dependencies
from unstructured.partition.utils.vlm_models.vlm_interface import VLMAgent
from PIL import Image as PILImage
from typing import TYPE_CHECKING
from unstructured_inference.inference.elements import TextRegions, TextRegion
from unstructured_inference.inference.layoutelement import LayoutElements


class VLMAgentQwen(VLMAgent):
    def __init__(self):
        self.agent_model, self.agent_processor = self.load_model_and_processor()


    def load_model_and_processor(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_CONF['model_name'],
                                                                   torch_dtype=torch.float16,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map=QWEN_CONF['device'])
        processor = AutoProcessor.from_pretrained(QWEN_CONF['model_name'])
        return model, processor

    def get_image_size(self, image:PILImage.Image) -> tuple[float, float]:
        img_width, img_height = image.size

        if img_width > QWEN_CONF['img_max_width'] or img_height > QWEN_CONF['img_max_height']:
            aspect_ratio = img_width / img_height
            new_width = min(QWEN_CONF['img_max_width'] , int(QWEN_CONF['img_max_height'] * aspect_ratio))
            new_height = min(QWEN_CONF['img_max_height'], int(QWEN_CONF['img_max_width'] / aspect_ratio))
            return (new_width, new_height)
        return img_width, img_height

    def get_updated_messages(self, filename: str, width: float, height: float, prompt: str) -> list[dict]:
        messages = copy.deepcopy(QWEN_CONF['messages'])
        messages[1]['content'][0]["image"] = filename
        messages[1]['content'][0]["resized_height"] = height
        messages[1]['content'][0]["resized_width"] = width
        messages[1]['content'][1]["text"] = prompt
        return messages

    def get_text_from_image(self, image: PILImage.Image) -> str:
        ocr_regions = self.get_layout_from_image(image)
        return "\n\n".join(ocr_regions.texts)

    def is_text_sorted(self):
        return False

    def get_vlm_response(self, image: PILImage.Image, filename: str, prompt: str) -> str:
        img_width, img_height = self.get_image_size(image)
        messages = self.get_updated_messages(filename, img_width, img_height, prompt)
        text = self.agent_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.agent_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(QWEN_CONF['device'])
        generated_ids = self.agent_model.generate(**inputs, max_new_tokens=QWEN_CONF['max_new_tokens'])
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_output = self.agent_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(raw_output[0])
        lines = raw_output[0].splitlines()
        for i, line in enumerate(lines):
          if line == '```json':
            json_data = "\n".join(lines[i+1: ])
            json_data = json_data.split("```")[0]
            break
        return json_data

    def post_process_raw_output(self, raw_output: str) -> dict:
        total_withdrawal = 0
        total_deposits = 0
        for details in ast.literal_eval(raw_output):
            for transaction in details["transactions"]:
                if 'inward' in transaction["transaction_type"].lower():
                    total_deposits += transaction['amount']
                elif 'to' in transaction["transaction_type"].lower():
                    total_withdrawal += transaction['amount']
                elif 'interest earned' in transaction["transaction_type"].lower():
                    total_deposits += transaction['amount']
            processed_output = copy.deepcopy(details)
            processed_output['total_withdrawal'] = total_withdrawal
            processed_output['total_deposits'] = total_deposits
        return processed_output


    def get_layout_from_image_without_bbox(self, image: PILImage.Image, filename: str, prompt:str) -> dict:
        vlm_output = self.get_vlm_response(image, filename, prompt)
        processed_output = self.post_process_raw_output(vlm_output)
        return processed_output

    def get_layout_from_image(self, image: PILImage.Image, filename: str, prompt:str) -> TextRegions:
        vlm_output = self.get_vlm_response(image, filename, prompt)
        text_regions = self.parse_raw_output(vlm_output)
        return text_regions
        

    @requires_dependencies("unstructured_inference")
    def get_layout_elements_from_image(self, image: PILImage.Image, filename: str, prompt:str) -> LayoutElements:
        ocr_regions = self.get_layout_from_image(image, filename, prompt)
        return LayoutElements(
            element_coords=ocr_regions.element_coords,
            texts=ocr_regions.texts,
            element_class_ids=np.zeros(ocr_regions.texts.shape),
            element_class_id_map={0: QWEN_CONF['unrecognized_text']},
        )

    @requires_dependencies("unstructured_inference")
    def parse_raw_output(self, raw_output : str) -> TextRegions:
        text_regions : list[TextRegion] = []

        # lines = raw_output.splitlines()
        # for i, line in enumerate(lines):
        #   if line == '```json':
        #     json_data = "\n".join(lines[i+1: ])
        #     json_data = json_data.split("```")[0]
        #     break

        for region in ast.literal_eval(raw_output):
            x1, y1, x2, y2 = region['bbox_2d']
            text = region['text_content']

            if not text:
                continue
            cleaned_text = text.strip()
            if cleaned_text:
                text_region = build_text_region_from_coords(
                    x1, y1, x2, y2, text=cleaned_text, source=Source.VLM_QWEN
                )
            text_regions.append(text_region)

        return TextRegions.from_list(text_regions)