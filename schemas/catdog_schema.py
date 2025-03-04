from pydantic import BaseModel

class CatDogResponse(BaseModel):
    """
    Response schema for catdog predictor

    Attributes:
        probs (list): List of probabilities for each class
        best_prob (float): Highest probability
        predicted_id (int): Predicted class ID
        predicted_class (str): Predicted class name
        predictor_name (str): Predictor
    """
    probs: list = []
    best_prob: float = -1.0
    predicted_id: int = -1
    predicted_class: str = ""
    predictor_name: str = ""
