
import torch
import torch.nn as nn
import torchvision.models as models
# from network.models.efficientnet.KBNetNEWy import cranet as mynet
from network.models.efficientnet.DRIMNET_IHR_V2 import cranet as mynet
# from network.models.efficientnet.convnext import convnext_base_kd as mynet

# from network.DINOV3.dinov3.vision_transformer import DinoVisionTransformer as mynet

class MainNet(nn.Module):
    def __init__(self, num_classes=None,device = None):
        print('num_classes:',num_classes)
        # print("ETSB")
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        # rgb_stream = mynet( pretrained=True, num_classes=2)
        rgb_stream = mynet(model_name1='efficientnet-b4',advprop1=True, num_classes1=2)
        self.rgb_extract_feature = rgb_stream
        # self.fc = nn.Linear(2120,num_classes)#3,4,5 b0       
    def forward(self, rgb_data, return_embedding=False, return_inc_embedding=False):
        output = self.rgb_extract_feature(
            rgb_data,
            return_embedding=return_embedding,
            return_inc_embedding=return_inc_embedding,
        )
        # output= self.fc(output)
        return output
    
