from app.services.node_service import fetch_from_node

async def call_node_service(data: dict):
    # Add any Python-side logic here before calling Node.js
    node_response = await fetch_from_node(data)
    return node_response