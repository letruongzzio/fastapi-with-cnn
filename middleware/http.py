import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from utils.logger import Logger

# Initialize Logger
LOGGER = Logger(name=__file__, log_file="http.log")

class LogMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests with processing time."""
    async def dispatch(self, request: Request, call_next: callable):
        """Middleware to log HTTP requests with processing time."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        LOGGER.log.info(
            f"{request.client.host} - \"{request.method} {request.url.path} "
            f"{request.scope['http_version']}\" {response.status_code} {process_time:.2f}s"
        )
        return response
