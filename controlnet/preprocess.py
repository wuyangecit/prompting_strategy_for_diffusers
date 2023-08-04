from PIL import Image
import io
from typing import Dict, Optional, Union
from controlnet_aux import (CannyDetector, HEDdetector,
                            LeresDetector, LineartAnimeDetector,
                            LineartDetector,MidasDetector,
                            OpenposeDetector, PidiNetDetector, ZoeDetector)


MODELS = {
    # checkpoint models
    'scribble_hed': {'className': 'HEDdetector', 'class': HEDdetector,'checkpoint': True},
    'scribble_hedsafe': {'className': 'HEDdetector', 'class': HEDdetector, 'checkpoint': True},
    'depth_midas': {'className': 'MidasDetector', 'class': MidasDetector, 'checkpoint': True},
    'openpose': {'className': 'OpenposeDetector', 'class': OpenposeDetector, 'checkpoint': True},
    'openpose_face': {'className': 'OpenposeDetector', 'class': OpenposeDetector, 'checkpoint': True},
    'openpose_faceonly': {'className': 'OpenposeDetector', 'class': OpenposeDetector, 'checkpoint': True},
    'openpose_full': {'className': 'OpenposeDetector', 'class': OpenposeDetector, 'checkpoint': True},
    'openpose_hand': {'className': 'OpenposeDetector', 'class': OpenposeDetector, 'checkpoint': True},
    'scribble_pidinet': {'className': 'PidiNetDetector', 'class': PidiNetDetector, 'checkpoint': True},
    'scribble_pidsafe': {'className': 'PidiNetDetector', 'class': PidiNetDetector, 'checkpoint': True},
    'lineart_coarse': {'className': 'LineartDetector', 'class': LineartDetector, 'checkpoint': True},
    'lineart_realistic': {'className': 'LineartDetector', 'class': LineartDetector, 'checkpoint': True},
    'lineart_anime': {'className': 'LineartAnimeDetector', 'class': LineartAnimeDetector, 'checkpoint': True},
    'depth_zoe': {'className': 'ZoeDetector', 'class': ZoeDetector, 'checkpoint': True},
    'depth_leres': {'className': 'LeresDetector', 'class': LeresDetector, 'checkpoint': True},
    'depth_leres++': {'className': 'LeresDetector', 'class': LeresDetector, 'checkpoint': True},
    # instantiate
    'canny': {'className': 'CannyDetector', 'class': CannyDetector, 'checkpoint': False},
}


MODEL_PARAMS = {
    'scribble_hed': {'scribble': True},
    'scribble_hedsafe': {'scribble': True, 'safe': True},
    'depth_midas': {},
    'openpose': {'include_body': True, 'include_hand': False, 'include_face': False},
    'openpose_face': {'include_body': True, 'include_hand': False, 'include_face': True},
    'openpose_faceonly': {'include_body': False, 'include_hand': False, 'include_face': True},
    'openpose_full': {'include_body': True, 'include_hand': True, 'include_face': True},
    'openpose_hand': {'include_body': False, 'include_hand': True, 'include_face': False},
    'scribble_pidinet': {'safe': False, 'scribble': True},
    'scribble_pidsafe': {'safe': True, 'scribble': True},
    'lineart_realistic': {'coarse': False},
    'lineart_coarse': {'coarse': True},
    'lineart_anime': {},
    'canny': {},
    'depth_zoe': {},
    'depth_leres': {'boost': False},
    'depth_leres++': {'boost': True},
}

class Preprocessor:
    def __init__(self, processor_list: list[str], model_path: str, params: Optional[Dict] = None) -> None:
        processor_models = {}
        processor_model_paras = {}
        self.model_path = model_path
        for processor_id in processor_list:
            if processor_id not in MODELS:
                continue
            processor_class_name = MODELS[processor_id]['className']
            if processor_class_name in processor_models:
                continue
            processor = self.load_processor(processor_id)
            processor_models[processor_class_name] = processor
            # load params
            processor_params = MODEL_PARAMS[processor_id]
            # update new params
            if params:
                processor_params.update(params)
            processor_model_paras[processor_id] = processor_params

        self.processor_model = processor_models
        self.processor_model_paras = processor_model_paras

    def load_processor(self, processor_id: str) -> 'Processor':
        '''
        Load processor from MODELS
        '''
        processor = MODELS[processor_id]['class']
        # check if the proecssor is a checkpoint model
        if MODELS[processor_id]['checkpoint']:
            processor = processor.from_pretrained(self.model_path)
        else:
            processor = processor()
        return processor

    def __call__(self, image: Union[Image.Image, bytes],
                    processor_id: str,to_pil: bool = True) -> Union[Image.Image, bytes]:
        '''
        目前只使用cpu进行预处理模型推理，后续如果有耗时长的模型，可以考虑使用gpu进行推理
        '''
        # check if processor_id is valid
        if processor_id not in MODELS:
            raise ValueError(f'Invalid processor_id: {processor_id}')
        # load unloaded processor
        if MODELS[processor_id]['className'] not in self.processor_model:
            processor_temp = self.load_processor(processor_id)
            self.processor_model[MODELS[processor_id]['className']] = processor_temp
            self.processor_model_paras[processor_id] = MODEL_PARAMS[processor_id]

        # check if bytes or PIL Image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        processor_class_name = MODELS[processor_id]['className']
        processor = self.processor_model[processor_class_name]
        processor_params = self.processor_model_paras.get(processor_id,MODEL_PARAMS[processor_id])
        processed_image = processor(image, **processor_params)

        if to_pil:
            return processed_image
        else:
            output_bytes = io.BytesIO()
            processed_image.save(output_bytes, format='JPEG')
            return output_bytes.getvalue()