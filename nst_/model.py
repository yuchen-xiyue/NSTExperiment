import torch
import torch.nn as nn
import torchvision.models as models

Tensor = torch.tensor


class NSTModel(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(NSTModel, self).__init__()
        if backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=True).features
            self.outputs = {
                'conv1_1': None, 'conv2_1': None, 'conv3_1': None,
                'conv4_1': None, 'conv4_2': None, 'conv5_1': None }
            # style
            self.backbone[ 1].register_forward_hook(self.outputs_hook('conv1_1'))
            self.backbone[ 6].register_forward_hook(self.outputs_hook('conv2_1'))
            self.backbone[11].register_forward_hook(self.outputs_hook('conv3_1'))
            self.backbone[18].register_forward_hook(self.outputs_hook('conv4_1'))
            self.backbone[25].register_forward_hook(self.outputs_hook('conv5_1'))
            # content
            self.backbone[20].register_forward_hook(self.outputs_hook('conv4_2'))

    def forward(self, X):
        X = self.backbone(X)

        style_outputs = [
            self.outputs['conv1_1'],
            self.outputs['conv2_1'],
            self.outputs['conv3_1'],
            self.outputs['conv4_1'],
            self.outputs['conv5_1']
        ]
        content_outputs = [self.outputs['conv4_2']]

        return style_outputs, content_outputs

    def outputs_hook(self, layer):
        def fn(_, __, output):
            self.outputs[layer] = output
        return fn