from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING


from unstructured.partition.utils.config import env_config

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from unstructured_inference.inference.elements import TextRegions
    from unstructured_inference.inference.layoutelement import LayoutElements


class VLMAgent(ABC):
    """Defines the Interface for Optical Character interface using Visual Language Models"""

    @classmethod
    def get_agent(cls) -> VLMAgent:
        vlm_agent_cls_qname = cls._get_vlm_agent_cls_qname()
        return cls.get_instance(vlm_agent_cls_qname)

    @staticmethod
    def _get_vlm_agent_cls_qname() -> str:
        vlm_agent_cls_qname = env_config.VLM_AGENT
        return vlm_agent_cls_qname

    @staticmethod
    @lru_cache()
    def get_instance(vlm_module: str) -> VLMAgent:
        module_name, class_name = vlm_module.rsplit('.', 1)
        module = import_module(module_name)
        loaded_class = getattr(module, class_name)
        return loaded_class()

    @abstractmethod
    def get_layout_from_image(self, image: PILImage.Image, filename: str) -> TextRegions:
        pass

    @abstractmethod
    def get_layout_elements_from_image(self, image: PILImage.Image) -> LayoutElements:
        pass

    @abstractmethod
    def get_text_from_image(self, image: PILImage.Image) -> str:
        pass

    @abstractmethod
    def is_text_sorted(self) -> bool:
        pass

