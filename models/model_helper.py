import copy
import importlib

import torch
import torch.nn as nn
from utils.misc_helper import to_device

def build_dcuad_model(cfg):
    """构建DCUAD模型"""
    # 使用原来的代码来构建所有组件（backbone，neck）
    model = ModelHelper(cfg.net)
    
    # 替换reconstruction部分为DCUAD
    reconstruction_config = cfg.net[2]
    if 'domain_discovery' in reconstruction_config.kwargs:
        num_domains = reconstruction_config.kwargs.domain_discovery.num_prototypes
        # 替换为DCUAD
        model.reconstruction = DCUAD(
            model.neck.get_outplanes(),
            model.neck.get_outstrides(),
            **reconstruction_config.kwargs,
            num_domains=num_domains
        )
    
    # 初始化并返回
    model.cuda()
    return model

class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        input = copy.copy(input)
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)
        for submodule in self.children():
            output = submodule(input)
            input.update(output)
        return input

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
