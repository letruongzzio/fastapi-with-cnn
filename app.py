import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from middleware import LogMiddleware, setup_cors
from routes.base import router

# Initialize FastAPI app
app = FastAPI()

# Add middleware and routes
app.add_middleware(LogMiddleware)
setup_cors(app)
app.include_router(router)
