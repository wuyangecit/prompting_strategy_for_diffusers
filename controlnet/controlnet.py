import os
from typing import Dict, Optional, Union
import torch
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel


class ControlNetLoader:
    def __init__(self, preLoad_controlNet_model: list[str], model_path: str, device='cpu', dtype=torch.float16) -> None:
        '''
        preload controlNet model
        '''
        self.model_path = model_path
        self.cn_model_cache = {}
        self.device = device
        self.dtype = dtype
        for cn_model_name in preLoad_controlNet_model:
            if cn_model_name not in self.cn_model_cache:
                controlnet_model = self.load_cn_model(cn_model_name)
                self.cn_model_cache[cn_model_name] = controlnet_model




    def load_cn_model(self, cn_model_name: str):
        '''
        Load cn_model from local_path
        '''
        model_dir = f'{self.model_path}/{cn_model_name}/'
        if not os.path.isdir(model_dir):
            raise ValueError(f'Invalid cn_model_name: {cn_model_name}')
        controlnet_model = ControlNetModel.from_pretrained(f'{self.model_path}/{cn_model_name}/', device='cpu', torch_dtype=self.dtype)
        return controlnet_model

    def __call__(self, cn_model_names: Union[str,list],):
        '''
        return cn_model
        '''
        controlNet_res = None
        if isinstance(cn_model_names, str):
            cn_model_names = [cn_model_names]

        if len(cn_model_names) == 1:
            for cn_model_name in cn_model_names:
                if cn_model_name not in self.cn_model_cache:
                    controlnet_model = self.load_cn_model(cn_model_name)
                    self.cn_model_cache[cn_model_name] = controlnet_model
                controlNet_res = self.cn_model_cache[cn_model_name]
        elif len(cn_model_names) > 1:
            controlnet_models = []
            for cn_model_name in cn_model_names:
                if cn_model_name not in self.cn_model_cache:
                    controlnet_model = self.load_cn_model(cn_model_name)
                    self.cn_model_cache[cn_model_name] = controlnet_model
                controlnet_models.append(self.cn_model_cache[cn_model_name])
            controlNet_res = MultiControlNetModel(controlnet_models)
        else:
            raise ValueError(f'Invalid cn_model_names: {cn_model_names}')
        return controlNet_res
