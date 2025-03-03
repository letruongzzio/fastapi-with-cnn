import sys
import os
import torch

ROOT_DIR = os.path.expanduser('~/fastapi-with-cnn')
sys.path.append(ROOT_DIR)

class CatDogDataConfig:
    """
    Configuration class for CatDogDataset
    """
    N_CLASSES = 2
    IMG_SIZE = 200
    ID2LABEL = {0: 'Cat', 1: 'Dog'}
    LABEL2ID = {'Cat': 0, 'Dog': 1}
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

class ModelConfig:
    """
    Configuration class for ResNet18 model
    """
    MODEL_NAME = 'resnet18'
    MODEL_WEIGHT = os.path.join(ROOT_DIR, 'models', 'weights', 'catvsdogs_classifier_resnet18.pt')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
