from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

# Pydantic Models for A2A Protocol
class MessagePart(BaseModel):
    kind: Literal["text", "data", "file"]
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    file_url: Optional[str] = None

class A2AMessage(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["user", "agent", "system"]
    parts: List[MessagePart]
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    taskId: Optional[str] = None

class MessageConfiguration(BaseModel):
    blocking: bool = True
    acceptedOutputModes: List[str] = ["text/plain", "image/png"]

class MessageParams(BaseModel):
    message: A2AMessage
    configuration: MessageConfiguration = Field(default_factory=MessageConfiguration)

class ExecuteParams(BaseModel):
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    messages: List[A2AMessage]

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str
    method: Literal["message/send", "execute"]
    params: MessageParams | ExecuteParams

class TaskStatus(BaseModel):
    state: Literal["working", "completed", "input-required", "failed"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message: Optional[A2AMessage] = None

class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: List[MessagePart]

class TaskResult(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    artifacts: List[Artifact] = []
    history: List[A2AMessage] = []
    kind: Literal["task"] = "task"

class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[TaskResult] = None
    error: Optional[Dict[str, Any]] = None

# FastAPI App
app = FastAPI(
    title="NASA Space Explorer Agent - A2A Compliant",
    description="A fully A2A protocol compliant NASA Astronomy Picture agent",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NASA API Configuration
NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

@app.post("/a2a/nasa")
async def a2a_endpoint(request: Request):
    """Fully A2A compliant endpoint"""
    try:
        body = await request.json()
        
        # Validate JSON-RPC 2.0 basics
        if body.get("jsonrpc") != "2.0" or "id" not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": body.get("id", "unknown"),
                    "error": {
                        "code": -32600,
                        "message": "Invalid Request"
                    }
                }
            )
        
        rpc_request = JSONRPCRequest(**body)
        
        # Process based on method
        if rpc_request.method == "message/send":
            result = await handle_message_send(rpc_request.params)
        elif rpc_request.method == "execute":
            result = await handle_execute(rpc_request.params)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": rpc_request.id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {rpc_request.method}"
                    }
                }
            )
        
        response = JSONRPCResponse(
            id=rpc_request.id,
            result=result
        )
        
        return response.model_dump()
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": body.get("id", "unknown") if "body" in locals() else "unknown",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": {"details": str(e)}
                }
            }
        )

async def handle_message_send(params: MessageParams):
    """Handle message/send method"""
     # DEBUG: Log the entire request
    print("=== DEBUG: Received A2A Request ===")
    print(f"Full params: {params}")
    print(f"Message parts: {params.message.parts}")
    user_message = params.message
    config = params.configuration
    
    # Extract user text from parts
    user_text = ""
    for part in user_message.parts:
        if part.kind == "text" and part.text:
            user_text = part.text.lower()
            break
    
    print(f"DEBUG: Final user_text: '{user_text}'")
    # Route based on user command
    if not user_text:
        user_text = "today's image"
        print("DEBUG: No user text, defaulting to 'today's image'")
        
    if "random" in user_text:
        nasa_data = await get_random_apod_data()
    elif "yesterday" in user_text:
        nasa_data = await get_yesterday_apod_data()
    elif "fact" in user_text:
        return await get_space_fact_response(user_message)
    else:
        nasa_data = await get_nasa_apod_data()
    
    # Create response message
    response_parts = [MessagePart(
        kind="text",
        text=format_nasa_response(nasa_data)
    )]
    
    # Add image as artifact if available
    artifacts = []
    if nasa_data.get('media_type') == 'image' and nasa_data.get('url'):
        artifacts.append(Artifact(
            name="nasa_image",
            parts=[MessagePart(kind="file", file_url=nasa_data['url'])]
        ))
    
    artifacts.append(Artifact(
        name="image_title",
        parts=[MessagePart(kind="text", text=nasa_data.get('title', 'NASA Astronomy Picture'))]
    ))
    
    response_message = A2AMessage(
        role="agent",
        parts=response_parts,
        messageId=str(uuid4()),
        taskId=user_message.taskId
    )
    
    # Build task result
    return TaskResult(
        id=user_message.taskId or str(uuid4()),
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed",
            message=response_message
        ),
        artifacts=artifacts,
        history=[user_message, response_message]
    )

async def handle_execute(params: ExecuteParams):
    """Handle execute method (for multiple messages)"""
    if not params.messages:
        # If no messages, create a default one
        user_message = A2AMessage(
            role="user",
            parts=[MessagePart(kind="text", text="today's image")]
        )
    else:
        user_message = params.messages[-1]
    
    return await handle_message_send(MessageParams(
        message=user_message,
        configuration=MessageConfiguration()
    ))

async def get_nasa_apod_data(date=None):
    """Fetch NASA APOD data"""
    try:
        url = f"{NASA_APOD_URL}?api_key={NASA_API_KEY}"
        if date:
            url += f"&date={date}"
            
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "title": "Error fetching NASA data",
            "explanation": "Could not retrieve the astronomy picture. Please try again later.",
            "url": "",
            "media_type": "image"
        }

async def get_random_apod_data():
    """Get random APOD from archive"""
    import random
    from datetime import datetime, timedelta
    
    # NASA APOD started June 16, 1995
    start_date = datetime(1995, 6, 16)
    end_date = datetime.now()
    random_date = start_date + timedelta(
        days=random.randint(0, (end_date - start_date).days)
    )
    
    return await get_nasa_apod_data(random_date.strftime("%Y-%m-%d"))

async def get_yesterday_apod_data():
    """Get yesterday's APOD"""
    from datetime import datetime, timedelta
    yesterday = datetime.now() - timedelta(days=1)
    return await get_nasa_apod_data(yesterday.strftime("%Y-%m-%d"))

async def get_space_fact_response(user_message: A2AMessage):
    """Return space fact response"""
    space_facts = [
        "A day on Mercury lasts 59 Earth days!",
        "Neptune's winds can reach 1,600 km/h - the fastest in the solar system!",
        "There are more stars in the universe than grains of sand on all Earth's beaches!",
        "A teaspoon of neutron star would weigh about 6 billion tons!",
        "Venus is the only planet that spins clockwise!",
        "The Sun makes up 99.86% of the mass in our solar system!",
    ]
    
    import random
    fact = random.choice(space_facts)
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=f"🌌 *Space Fact* 🌌\n\n{fact}")],
        messageId=str(uuid4()),
        taskId=user_message.taskId
    )
    
    return TaskResult(
        id=user_message.taskId or str(uuid4()),
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed",
            message=response_message
        ),
        artifacts=[
            Artifact(
                name="space_fact",
                parts=[MessagePart(kind="text", text=fact)]
            )
        ],
        history=[user_message, response_message]
    )

def format_nasa_response(data):
    """Format NASA data into nice response"""
    title = data.get('title', 'Unknown Title')
    explanation = data.get('explanation', 'No description available.')
    date = data.get('date', 'Unknown date')
    
    # Truncate long explanations
    if len(explanation) > 1000:
        explanation = explanation[:1000] + "..."
    
    return f"""🌌 *{title}* 🌌

{explanation}

*Details:*
📅 Date: {date}
🖼️ Media Type: {data.get('media_type', 'image')}

*Explore the cosmos!* 🚀"""

@app.get("/")
async def root():
    return {
        "message": "NASA Space Explorer Agent is running!",
        "status": "healthy", 
        "version": "2.0.0",
        "protocol": "A2A Compliant",
        "endpoints": {
            "POST /a2a/nasa": "Main A2A endpoint",
            "GET /health": "Health check",
            "GET /test": "Test NASA connection"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NASA Space Explorer Agent"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify NASA API connection"""
    try:
        data = await get_nasa_apod_data()
        return {
            "nasa_api_status": "connected",
            "agent_status": "healthy",
            "sample_data": {
                "title": data.get('title'),
                "media_type": data.get('media_type')
            }
        }
    except Exception as e:
        return {"nasa_api_status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)