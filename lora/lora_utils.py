import math
import safetensors
import torch
from diffusers import DiffusionPipeline


class LoRAModuleWeight():
    def __init__(self, lora_name, module_dict, state_dict, multiplier, device, dtype):
        super().__init__()
        self.multiplier = multiplier
        self.new_lora_weight = {}
        self.lora_name = lora_name
        self.device = device

        # Create LoRAModule from state_dict information
        with torch.no_grad():
            for key, value in state_dict.items():
                if "lora_down" in key and key.split(".")[0] in module_dict:
                    lora_layer_name, suffix = key.split('.', 1)
                    lora_name_alpha = key.split(".")[0] + '.alpha'
                    alpha = None
                    if lora_name_alpha in state_dict:
                        alpha = state_dict[lora_name_alpha].item()
                    module_class_name = module_dict[lora_layer_name]
                    # 计算权重
                    down_weight = value.to(device, dtype=dtype)
                    up_weight = state_dict[key.replace('lora_down.weight', 'lora_up.weight')].to(device, dtype=dtype)

                    dim = down_weight.size()[0] if down_weight != None else 4

                    if alpha is None or alpha == 0:
                        alpha = 1.0

                    scale = alpha / dim

                    if module_class_name == "Linear":
                        # linear
                        new_weight = multiplier * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        new_weight = (
                                multiplier
                                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(
                            2).unsqueeze(3)
                                * scale
                        )
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2,
                                                                                                                3)
                        new_weight = multiplier * conved * scale

                    self.new_lora_weight[lora_layer_name] = new_weight.to("cpu")

    def apply_role_to_hooks(self, hooks, multiplier=1.0):
        for layer_name, weight in self.new_lora_weight.items():
            hook = hooks[layer_name]
            hook.append_lora(weight.to(self.device), multiplier)
            # hook.load_lora()


class LoRAHook(torch.nn.Module):
    """
    replaces forward method of the original Linear,
    instead of replacing the original Linear module.
    """

    def __init__(self, device):
        super().__init__()
        self.lora_modules = []
        self.device = device

    def install(self, orig_module):
        assert not hasattr(self, "orig_module")
        orig_module.weight.requires_grad = False
        #         print(orig_module.weight)
        self.orig_module = orig_module
        self.orig_module_weight = orig_module.weight.to("cpu", copy=True)

    def append_lora(self, weight, multiplier):
        self.lora_modules.append((weight, multiplier))

    def load_lora(self):
        with torch.no_grad():
            self.orig_module.weight.copy_(self.orig_module_weight)
        for weight, multiplier in self.lora_modules:
            if multiplier == 1.0:
                self.orig_module.weight += weight
            else:
                self.orig_module.weight += weight * multiplier

    def clear_lora(self):
        if len(self.lora_modules) > 0:
            with torch.no_grad():
                self.lora_modules.clear()
                self.orig_module.weight.copy_(self.orig_module_weight)


class LoRAHookInjector(object):
    def __init__(self):
        super().__init__()
        self.hooks = {}
        self.device = None
        self.dtype = None

    def _get_target_modules(self, root_module, prefix, target_replace_modules):
        target_modules = []
        for name, module in root_module.named_modules():
            if (
                    module.__class__.__name__ in target_replace_modules
                    and not "transformer_blocks" in name
            ):  # to adapt latest diffusers:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    if is_linear or is_conv2d:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        target_modules.append((lora_name, child_module))
        return target_modules

    def install_hooks(self, pipe):
        """Install LoRAHook to the pipe."""
        assert len(self.hooks) == 0
        text_encoder_targets = self._get_target_modules(
            pipe.text_encoder, "lora_te", ["CLIPAttention", "CLIPMLP"]
        )
        unet_targets = self._get_target_modules(
            pipe.unet, "lora_unet", ["Transformer2DModel", "Attention"]
        )
        for name, target_module in text_encoder_targets + unet_targets:
            hook = LoRAHook(pipe.device)
            hook.install(target_module)
            self.hooks[name] = hook

        self.device = pipe.device
        self.dtype = pipe.unet.dtype

    def apply_lora(self):
        """Load LoRA weights and apply LoRA to the pipe."""
        if self.hooks != None and len(self.hooks) > 0:
            for name, hook in self.hooks.items():
                hook.load_lora()
            return 0
        else:
            return -1

    def load_lora(self, LoraWeight, multiplier):
        """"add lora weight to the every hook"""
        LoraWeight.apply_role_to_hooks(self.hooks, multiplier)

    def remove_lora(self, LoraWeight):
        """Remove the single LoRA from the pipe."""
        LoraWeight.remove_from_hooks(self.hooks)

    def clear_lora(self):
        if self.hooks != None and len(self.hooks) > 0:
            for name, hook in self.hooks.items():
                hook.clear_lora()



