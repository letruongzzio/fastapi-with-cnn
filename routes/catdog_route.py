import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import File, UploadFile, APIRouter
from schemas.catdog_schema import CatDogResponse
from config.catdog_cfg import ModelConfig
from models.catdog_predictor import Predictor

# Initialize API router
router = APIRouter()

# Initialize Predictor instance
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight=ModelConfig.MODEL_WEIGHT,
    device=ModelConfig.DEVICE
)

@router.post("/predict")
async def predict(file_upload: UploadFile = File(...)):
    """Handles file upload and returns model prediction."""
    response = await predictor.predict(file_upload.file)
    return CatDogResponse(**response)