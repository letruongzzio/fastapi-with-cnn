"""
In case you have multiple routers, we can create a .py file to import all routers from individual files and combine them into a single router. In this case, however, we will only import one router.
"""

from fastapi import APIRouter
from routes.catdog_route import router as catdog_cls_route

router = APIRouter()
router.include_router(catdog_cls_route, prefix="/catdog_classification")