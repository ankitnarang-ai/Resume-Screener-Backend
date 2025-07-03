from fastapi import APIRouter, HTTPException
from app.controllers.node_controller import call_node_service

router = APIRouter()

@router.post("/call-node")
async def call_node_endpoint(data: dict):
    try:
        response = await call_node_service(data)
        return {"status": "success", "response_from_node": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))