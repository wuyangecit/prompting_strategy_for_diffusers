# prompting_strategy_for_diffusers

## 1.embedding
### feature
* prompt supports a limit of more than 77 tokens
* support texual_inversion
* Support weight，比如(1girl:1.2),[1boy]
* Similar to a1111 embedding

## 2. lora
### feature
* Support kohya style LORA
* The lora model can be dynamically switched

## 3. upscale
### feature
* Only supports RealESRGAN_x4plus, RealESRGAN_x4plus_anime_6B
* The upscale process is from t2i to upscale then to i2i

## 4. controlNet
### feature
* Controlnet model can be dynamically switched
* Preprocessing model can be dynamically switched
* Controlnet supports t2i, i2i, lora





