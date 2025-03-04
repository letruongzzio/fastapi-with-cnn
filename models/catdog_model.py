import torch.nn as nn
from torchvision.models import resnet18
from config.catdog_cfg import CatDogDataConfig


class CatDogModel(nn.Module):
    """
    CatDogModel is a neural network model for classifying images of cats and dogs.

    Attributes:
        model (torchvision.models.ResNet): Pretrained ResNet-18 model.
        backbone (torch.nn.Sequential): Sequential model excluding the final fully connected layer of ResNet-18.

    Methods:
        __init__(num_classes: int = NUM_CLASSES):
            Initializes the CatDogModel with a specified number of output classes.
            Args:
                num_classes (int): Number of output classes for the classification task. Default is NUM_CLASSES.
        
        forward(x):
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor representing a batch of images.
            Returns:
                torch.Tensor: Output tensor representing the class scores for each input image.
    """
    def __init__(self, num_classes: int = CatDogDataConfig.N_CLASSES):
        super(CatDogModel, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        unfreeze = ['layer4', 'fc']
        for layer_name, layer in self.model.named_parameters():
            for name in unfreeze:
                if name in layer_name:
                    layer.requires_grad = True
                    break
                else:
                    layer.requires_grad = False
    def forward(self, x):
        return self.model(x)
