from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
from PIL import Image
import numpy as np
import os

class Upscale:
    models = {}
    def __init__(self, model_path_dir:str):
        self.model_path_dir = model_path_dir

    def upscale(
        self,
        model_name,
        img: Image.Image,
        scale_factor: float = 4,
    ) -> Image.Image:
        # upscale
        upsampler = self.models.get(model_name, None)
        if upsampler is None:
            return None
        torch.cuda.empty_cache()
        img = np.array(img)
        img_res = upsampler.enhance(img, outscale=scale_factor)[0]
        image = Image.fromarray(np.uint8(img_res))
        torch.cuda.empty_cache()
        return image

    def upload_model_from_dir(self):
        for file_name in os.listdir(self.model_path_dir):
            model_name, ext = os.path.splitext(file_name)
            try:
                model = self.upload_upscale_model(model_name, self.model_path_dir)
            except NotImplementedError:
                continue
            self.models[model_name] = model

    def upload_upscale_model(
            self,
            model_name: str,
            model_path_dir:str,
            half_precision: bool = False,
            tile: int = 0,
            tile_pad: int = 10,
            pre_pad: int = 0,
    ):
        if model_name == "RealESRGAN_x4plus":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = os.path.join(model_path_dir,'RealESRGAN_x4plus.pth')

        elif model_name == "RealESRGAN_x4plus_anime_6B":
            upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = os.path.join(model_path_dir,'RealESRGAN_x4plus_anime_6B.pth')
        else:
            raise NotImplementedError("Model name not supported")

        # declare the upscaler
        upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=upscale_model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half_precision,
        gpu_id=None
        )

        return upsampler
