import torch
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
        self.model = resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
