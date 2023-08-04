from diffusers import DiffusionPipeline,StableDiffusionControlNetPipeline,ControlNetModel,StableDiffusionControlNetImg2ImgPipeline
import torch

from .preprocess import Preprocessor
from .controlnet import ControlNetLoader

from typing import List, Optional, Tuple, Union
from PIL import Image
import time
import gc
import random
import json
from utils.log_config import get_logger
log = get_logger('cn_process')


class ControlNetProcess:
    def __init__(self, generator: DiffusionPipeline,
                 controlnet_names:list[str],
                 preprocess_names:list[str],
                 controlnet_model_path:str,
                 preprocess_model_path:str,
                 preprocess_params=None,
                 device='cpu', dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.controlnet_model_cache = {}
        #加载模型加载器
        self.controlnet_loader = ControlNetLoader(controlnet_names, controlnet_model_path)
        #加载预处理模型
        self.preprocessor = Preprocessor(preprocess_names, preprocess_model_path, preprocess_params)
        #加载controlnet
        controlnet_model = self.controlnet_loader(controlnet_names)
        #加载pipeline
        self.pipelineT2I = StableDiffusionControlNetPipeline(vae=generator.vae,
                                                          text_encoder=generator.text_encoder,
                                                          tokenizer=generator.tokenizer,
                                                          unet=generator.unet,
                                                          controlnet=controlnet_model,
                                                          scheduler=generator.scheduler,
                                                          safety_checker=None,
                                                          feature_extractor=None,
                                                          requires_safety_checker=False
                                                          )
        self.pipelineT2I.enable_xformers_memory_efficient_attention()
        self.install_lora_hook(generator, self.pipelineT2I)

        self.pipelineI2I = StableDiffusionControlNetImg2ImgPipeline(vae=generator.vae,
                                                          text_encoder=generator.text_encoder,
                                                          tokenizer=generator.tokenizer,
                                                          unet=generator.unet,
                                                          controlnet=controlnet_model,
                                                          scheduler=generator.scheduler,
                                                          safety_checker=None,
                                                          feature_extractor=None,
                                                          requires_safety_checker=False
                                                          )
        self.pipelineI2I.enable_xformers_memory_efficient_attention()
        self.install_lora_hook(generator, self.pipelineI2I)

    def install_lora_hook(self,pipe: DiffusionPipeline, pipe_target):
        """Install LoRAHook to the pipe."""
        if hasattr(pipe, "lora_injector"):
            pipe_target.lora_injector = pipe.lora_injector
            pipe_target.load_lora = pipe.load_lora
            pipe_target.apply_lora = pipe.apply_lora
            pipe_target.clear_lora = pipe.clear_lora
        else:
            return

    def __call__(self,
                 prompt,
                 prompt_embeds,
                 negative_prompt,
                 negative_prompt_embeds,
                 apply_lora_list,
                 controlnet_info:Union[
                    dict,
                    List[dict],
                 ] = None,
                 init_image=None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 seed: Optional[int] = -1,
                 strength: float = 0.8,
                 guidance_scale: float = 7.5,
                 num_inference_steps=25,
                 num_images_per_prompt: Optional[int] = 1,
                 guess_mode: bool = False,
                 task_id=None
                 ):
        try:
            #---多重controlnet判断---#
            controlnet_res_info = {}
            start_time = time.time()
            if controlnet_info is None:
                raise Exception("controlnet is none,please use other pipeline")
            isMutipleControlNet = False
            isI2I = False
            #---判断是否为图生图---#
            if init_image is not None:
                isI2I = True
            #---判断是否为多个controlnet模块---#
            if isinstance(controlnet_info, list):
                if len(controlnet_info) > 1:
                    isMutipleControlNet = True

            controlnet_res_info["isI2I"] = isI2I

            #---预处理图片---#
            controlnet_model_name = None
            controlnet_image = None
            controlnet_Scale = None
            if isMutipleControlNet:
                temp_img_list = []
                temp_scale_list = []
                temp_model_name_list = []
                for index,controlnet_info_temp in enumerate(controlnet_info):
                    temp_model_name_list.append(controlnet_info_temp.get("cn_model_name",""))
                    temp_scale_list.append(float(controlnet_info_temp.get("cn_model_scale",1.0)))
                    if controlnet_info_temp["cn_image"] is not None:
                        if controlnet_info_temp["cn_pre_model_name"] == "":
                            temp_img_list.append(controlnet_info_temp["cn_image"])
                        else:
                            temp_img_list.append(self.preprocessor(controlnet_info_temp["cn_image"], controlnet_info_temp["cn_pre_model_name"]))
                    controlnet_image = temp_img_list
                    controlnet_Scale = temp_scale_list
                    controlnet_model_name = temp_model_name_list
            else:
                controlnet_info = controlnet_info[0]
                controlnet_model_name = controlnet_info.get("cn_model_name","")
                controlnet_Scale = float(controlnet_info.get("cn_model_scale",1.0))
                controlnet_image = controlnet_info.get("cn_image",None)
                if controlnet_image is not None:
                    if controlnet_info["cn_pre_model_name"] != "":
                        controlnet_image = self.preprocessor(controlnet_image, controlnet_info["cn_pre_model_name"])
            end_preprocess_time = time.time()
            controlnet_res_info["controlnet_models"] = controlnet_model_name
            controlnet_res_info["controlnet_Scale"] = controlnet_Scale

            #---加载pipeline---#
            if not isI2I:
                pipeline = self.pipelineT2I
            else:
                pipeline = self.pipelineI2I
            #---加载controlnet模型---#
            controlnet_model = self.controlnet_loader(controlnet_model_name)
            controlnet_model.to(self.device)
            pipeline.controlnet = None
            self.unload_controlnet_model()
            pipeline.controlnet = controlnet_model

            end_load_controlnet_time = time.time()
            # ---t2i/i2i模型运行---#
            # 提取seed
            if seed == -1:
                seed = int(random.randrange(4294967294))
            Generator = [torch.Generator(device="cuda").manual_seed(i) for i in
                         range(seed, seed + num_images_per_prompt)]
            if not isI2I:
                out = pipeline(prompt_embeds=prompt_embeds,
                               negative_prompt_embeds=negative_prompt_embeds,
                               image=controlnet_image,
                               height=height,
                               width=width,
                               guidance_scale=guidance_scale,
                               num_inference_steps=num_inference_steps,
                               num_images_per_prompt=num_images_per_prompt,
                               controlnet_conditioning_scale=controlnet_Scale,
                               guess_mode=guess_mode,
                               generator=Generator,
                               ).images
                image_info = self.get_image_info_controlnet(isI2I,prompt, width, height, num_inference_steps, guidance_scale,
                                            num_images_per_prompt, seed, controlnet_res_info,
                                            task_id=task_id,
                                            negative_prompt=negative_prompt,
                                            apply_lora_list=apply_lora_list)
            else:
                out = pipeline(prompt_embeds=prompt_embeds,
                               negative_prompt_embeds=negative_prompt_embeds,
                               image=init_image,
                               control_image=controlnet_image,
                               height=height,
                               width=width,
                               strength=strength,
                               guidance_scale=guidance_scale,
                               num_inference_steps=num_inference_steps,
                               num_images_per_prompt=num_images_per_prompt,
                               controlnet_conditioning_scale=controlnet_Scale,
                               guess_mode=guess_mode,
                               generator=Generator,
                               ).images
                image_info = self.get_image_info_controlnet(isI2I, prompt, width, height, num_inference_steps,
                                                            guidance_scale,num_images_per_prompt,
                                                            seed, controlnet_res_info,
                                                            task_id=task_id,
                                                            negative_prompt=negative_prompt,
                                                            apply_lora_list=apply_lora_list)

            end_generate_time = time.time()
            #controlnet模型卸载
            pipeline.controlnet.to('cpu')
            pipeline.controlnet = None
            self.unload_controlnet_model()
            end_unload_controlnet_time = time.time()
            log.info(
                f'controlnet inner time consume: {round((end_unload_controlnet_time - start_time), 2)}|preprocess: {round((end_preprocess_time - start_time), 2)}|load_controlnet: {round((end_load_controlnet_time - end_preprocess_time), 2)}|generate_image: {round((end_generate_time - end_load_controlnet_time), 2)}|unload_controlnet: {round((end_unload_controlnet_time - end_generate_time), 2)}')
            return out,image_info

        except Exception as e:
            log.error(f'controlnet inner error: {e}', exc_info=True)
            return None,None

    def unload_controlnet_model(self):
        gc.collect()
        self.torch_gc()

    def torch_gc(self):
        if torch.cuda.is_available():
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def get_image_info_controlnet(self,isI2I, prompt, width, height, num_inference_steps, guidance_scale, num_images_per_prompt,
                       seed, controlnet_res_info, denoise_strength=0.0, negative_prompt='', apply_lora_list=None, task_id=None):
        image_info_res = []
        for i in range(num_images_per_prompt):
            info_dict = dict()
            info_dict['prompt'] = prompt
            info_dict['negative_prompt'] = negative_prompt
            info_dict['controlnet_info'] = controlnet_res_info
            info_dict['width'] = width
            info_dict['height'] = height
            info_dict['num_inference_steps'] = num_inference_steps
            info_dict['guidance_scale'] = guidance_scale
            info_dict['seed'] = seed + i
            if isI2I:
                info_dict['denoise_strength'] = denoise_strength if denoise_strength else 0
            info_dict['loras'] = apply_lora_list if apply_lora_list else []
            if task_id is not None:
                info_dict['task_id'] = task_id
            image_info_res.append(json.dumps(info_dict))
        return image_info_res
