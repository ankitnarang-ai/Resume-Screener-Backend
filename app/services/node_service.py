import httpx  # Requires: pip install httpx

NODE_SERVICE_URL = "http://localhost:3000/api/process"  # Your Node.js endpoint

async def fetch_from_node(data: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(NODE_SERVICE_URL, json=data)
        response.raise_for_status()  # Raise errors for 4XX/5XX
        return response.json()