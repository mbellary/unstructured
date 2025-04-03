import copy
import numpy as np
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords
from unstructured.partition.utils.constants import Source
from unstructured.utils import requires_dependencies
from vlm_interface import VLMAgent
from PIL import Image as PILImage

if TYPE_CHECKING:
    from unstructured_inference.inference.elements import TextRegions, TextRegion
    from unstructured_inference.inference.layoutelement import LayoutElements

QWEN_MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct'
QWEN_MESSAGES = [
            {
            "role": "system",
            "content": "You are an expert at extracting structured text from image documents."
            },
            {
              "role": "user",
              "content": [
                          {
                            "type": "image",
                            "image": "",
                            "resized_height": "",
                            "resized_width": ""
                          },
                          {
                            "type": "text",
                            "text": "Spotting all the text in the image with line-level, and output in JSON format."
                          }
              ]
            }
        ]
MAX_WIDTH = 1250
MAX_HEIGHT = 1750
UNCATEGORIZED_TEXT = "UncategorizedText"


class VLMAgentQwen(VLMAgent):
    def __init__(self):
        self.agent_model, self.agent_processor = self.load_model_and_processor()


    def load_model_and_processor(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        return model, processor

    def get_image_size(self, image:PILImage.Image) -> tuple[float, float]:
        img_width, img_height = image.size

        if img_width > MAX_WIDTH or img_height > MAX_HEIGHT:
            aspect_ratio = img_width / img_height
            new_width = min(MAX_WIDTH, int(MAX_HEIGHT * aspect_ratio))
            new_height = min(MAX_HEIGHT, int(MAX_WIDTH / aspect_ratio))
            return (new_width, new_height)
        return img_width, img_height

    def get_updated_messages(self, filename: str, width: float, height: float) -> list[dict]:
        messages = copy.deepcopy(QWEN_MESSAGES)
        messages[1]['content'][0]["image"] = filename
        messages[1]['content'][0]["resized_height"] = height
        messages[1]['content'][0]["resized_width"] = width
        return messages

    def get_text_from_image(self, image: PILImage.Image) -> str:
        ocr_regions = self.get_layout_from_image(image)
        return "\n\n".join(ocr_regions.texts)

    def is_text_sorted(self):
        return False

    def get_layout_from_image(self, image: PILImage.Image, filename: str) -> TextRegions:
        img_width, img_height = self.get_image_size(image)
        messages = self.get_updated_messages(filename, img_width, img_height)
        text = self.agent_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.agent_model(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        generated_ids = self.agent_model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw_output = self.agent_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(raw_output[0])
        text_regions = self.parse_raw_output(raw_output[0])

        return text_regions
        

    @requires_dependencies("unstructured_inference")
    def get_layout_elements_from_image(self, image: PILImage.Image, filename: str) -> LayoutElements:
        ocr_regions = self.get_layout_from_image(image, filename)
        return LayoutElements(
            element_coords=ocr_regions.element_coords,
            texts=ocr_regions.texts,
            element_class_ids=np.zeros(ocr_regions.texts.shape),
            element_class_id_map={0: UNCATEGORIZED_TEXT},
        )

    @requires_dependencies("unstructured_inference")
    def parse_raw_output(self, raw_output : list[dict]) -> TextRegions:
        text_regions : list[TextRegion] = []

        for region in raw_output:
            x1, y1, x2, y2 = region['bbox_2d']
            text = region['text_content']

            if not text:
                continue
            cleaned_text = text.strip()
            if cleaned_text:
                text_region = build_text_region_from_coords(
                    x1, y1, x2, y2, text=cleaned_text, source=Source.VLM_QWEN
                )
            print(f'text region : {text_region}')
            text_regions.append(text_region)
            print(f'text regions : {text_regions}')

        return TextRegions.from_list(text_regions)