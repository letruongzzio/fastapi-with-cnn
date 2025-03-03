from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def setup_cors(app: FastAPI):
    """
    Setup CORS middleware for the FastAPI app.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # "http://localhost", "http://localhost:8080", "http://localhost:3000" etc.
        allow_credentials=True, # Set it to True if your frontend app is setting cookies.
        allow_methods=["*"], # "GET", "POST", "PUT", "DELETE", "OPTIONS" etc.
        allow_headers=["*"], # "Content-Type", "Authorization" etc.
    )
