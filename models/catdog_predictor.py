import sys
import os
import torch
import torchvision
import logging
from PIL import Image
from torch.nn import functional as F
from config.catdog_cfg import CatDogDataConfig
from utils.logger import Logger
from .catdog_model import CatDogModel

# Initialize Logger
LOGGER = Logger(name=__file__, log_file="predictor.log")
LOGGER.log.info("Starting Model Serving")

class Predictor:
    """
    Predictor is a class for making predictions using a trained model.

    Attributes:
        model_name (str): The name of the model.
        model_weight (str): The path to the model's weights.
        device (str): The device to run the model on (default is "cpu").
    Methods:
        __init__(model_name: str, model_weight: str, device: str = "cpu"):
            Initializes the Predictor with the given model name, weights, and device.
        predict(image):
            Asynchronously processes an input image and returns a prediction.
        model_inference(input):
            Asynchronously performs inference using the loaded model.
        load_model():
            Loads the model and its weights.
        create_transform():
            Creates the necessary image transformations.
        output2pred(output):
            Converts model output to prediction results.
    """
    def __init__(self, model_name: str, model_weight: str, device: str = "cpu"):
        self.model_name = model_name
        self.model_weight = model_weight
        self.device = device
        self.load_model()
        self.create_transform()
    
    async def predict(self, image):
        """Processes an input image and returns a prediction."""
        pil_img = Image.open(image)
        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")
        
        transformed_image = self.transforms_(pil_img).unsqueeze(0)
        output = await self.model_inference(transformed_image)
        probs, best_prob, predicted_id, predicted_class = self.output2pred(output)
        
        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)
        
        torch.cuda.empty_cache()
        
        return {
            "probs": probs,
            "best_prob": best_prob,
            "predicted_id": predicted_id,
            "predicted_class": predicted_class,
            "predictor_name": self.model_name
        }
    
    async def model_inference(self, input_tensor):
        """Performs inference using the loaded model."""
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(input_tensor).cpu()
        return output
    
    def load_model(self):
        """Loads the model and its weights."""
        try:
            model = CatDogModel(CatDogDataConfig.N_CLASSES)
            model.load_state_dict(torch.load(self.model_weight, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.loaded_model = model
        except (OSError, RuntimeError) as e:
            LOGGER.log.error("Load model failed")
            LOGGER.log.error("Error: %s", e)
            self.loaded_model = None
    
    def create_transform(self):
        """Creates the necessary image transformations."""
        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((CatDogDataConfig.IMG_SIZE, CatDogDataConfig.IMG_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=CatDogDataConfig.NORMALIZE_MEAN, std=CatDogDataConfig.NORMALIZE_STD)
        ])
    
    def output2pred(self, output):
        """Converts model output to prediction results."""
        probabilities = F.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = CatDogDataConfig.ID2LABEL[predicted_id]
        return probabilities.squeeze().tolist(), round(best_prob, 6), predicted_id, predicted_class
