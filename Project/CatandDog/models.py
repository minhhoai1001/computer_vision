import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

def resnet50(pretrained=True, requires_grad=False):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # either freeze or train the hidden layer parameters
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    model.fc = nn.Linear(2048, 1)
    return model

def efficientNet(name='efficientnet-b0', num_output = 1, pretrained=True):
    if pretrained:
        model = EfficientNet.from_pretrained(name)
    else: 
        model = EfficientNet.from_name(name)
    feature = model._fc.in_features
    model._fc = nn.Linear(feature, num_output)
    return model

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model