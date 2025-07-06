from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14@336px" : 768

}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, opt=None):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear(CHANNELS[name], num_classes )
        self.opt = opt
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)

