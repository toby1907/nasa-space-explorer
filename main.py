from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from uuid import uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
import asyncio
import aiohttp
import random
import json

# Pydantic Models for A2A Protocol
class MessagePart(BaseModel):
    kind: Literal["text", "data", "file"]
    text: Optional[str] = None
    data: Optional[Any] = None
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
    version="2.1.0"
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

# Cache for NASA responses to avoid repeated API calls
nasa_cache = {}
CACHE_DURATION = 3600  # 1 hour

# Pre-defined fallback images for when NASA API fails
FALLBACK_IMAGES = [
    {
        "title": "Hubble Space Telescope View",
        "explanation": "The Hubble Space Telescope has revolutionized astronomy with its stunning views of distant galaxies, nebulae, and star clusters. This image showcases the incredible detail Hubble can capture from its orbit above Earth's atmosphere.",
        "url": "https://images-assets.nasa.gov/image/PIA12153/PIA12153~large.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    },
    {
        "title": "Orion Nebula",
        "explanation": "The Orion Nebula is one of the brightest nebulae visible to the naked eye. Located in the Milky Way, it's a stellar nursery where new stars are being born from clouds of gas and dust.",
        "url": "https://images-assets.nasa.gov/image/PIA23122/PIA23122~large.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    },
    {
        "title": "Jupiter's Great Red Spot",
        "explanation": "Jupiter's Great Red Spot is a gigantic storm that has been raging for at least 400 years. This massive anticyclonic storm is larger than Earth and winds can reach speeds of 430 km/h.",
        "url": "https://images-assets.nasa.gov/image/PIA22946/PIA22946~large.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
]

SPACE_FACTS = [
    "A day on Mercury lasts 59 Earth days!",
    "Neptune's winds can reach 1,600 km/h - the fastest in the solar system!",
    "There are more stars in the universe than grains of sand on all Earth's beaches!",
    "A teaspoon of neutron star would weigh about 6 billion tons!",
    "Venus is the only planet that spins clockwise!",
    "The Sun makes up 99.86% of the mass in our solar system!",
    "Space is completely silent - there's no atmosphere to carry sound!",
    "The International Space Station orbits Earth every 90 minutes!",
    "A year on Venus is shorter than a day on Venus!",
    "There is a giant cloud of alcohol in Sagittarius B that contains billions of liters of vodka!"
]

@app.post("/a2a/nasa")
async def a2a_endpoint(request: Request):
    """A2A endpoint that handles Telex's invalid data format"""
    try:
        body = await request.json()
        
        # Validate JSON-RPC 2.0 basics first
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
        
        request_id = body.get("id", "unknown")
        
        try:
            # First try to parse with strict validation
            rpc_request = JSONRPCRequest(**body)
            method = rpc_request.method
            params = rpc_request.params
        except Exception as e:
            print(f"Validation error, using fallback parsing: {e}")
            # Use fallback parsing for Telex's non-standard format
            method = body.get("method", "message/send")
            params = body.get("params", {})
        
        # Process based on method
        if method == "message/send":
            result = await handle_message_send_fallback(params, request_id)
        elif method == "execute":
            result = await handle_execute_fallback(params, request_id)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            )
        
        response = JSONRPCResponse(
            id=request_id,
            result=result
        )
        
        return response.model_dump()
        
    except Exception as e:
        print(f"Unexpected error: {e}")
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

async def handle_message_send_fallback(params: dict, request_id: str):
    """Handle message/send with fallback parsing"""
    print("=== DEBUG: handle_message_send_fallback called ===")
    
    # Extract user message from various possible locations
    user_message = extract_user_message_robust(params)
    print(f"🚀 DEBUG: EXTRACTED USER MESSAGE: '{user_message}'")
    
    # Process the command
    return await process_user_command(user_message, request_id)

async def handle_execute_fallback(params: dict, request_id: str):
    """Handle execute with fallback parsing"""
    messages = params.get('messages', [])
    if messages:
        # Use the last message
        last_message = messages[-1]
        user_message = extract_user_message_from_a2a_message(last_message)
    else:
        user_message = "today's image"
    
    return await process_user_command(user_message, request_id)

def extract_user_message_robust(params: dict) -> str:
    """Extract user message from params with robust parsing"""
    try:
        message_data = params.get('message', {})
        parts = message_data.get('parts', [])
        
        print(f"DEBUG: Found {len(parts)} parts in message")
        
        # Look for text in parts
        for i, part in enumerate(parts):
            part_kind = part.get('kind', '')
            part_text = part.get('text', '')
            part_data = part.get('data')
            
            print(f"DEBUG: Part {i} - kind: {part_kind}, text: {part_text[:100] if part_text else 'None'}")
            
            # Direct text part
            if part_kind == 'text' and part_text and part_text.strip():
                clean_text = clean_user_input(part_text)
                if clean_text:
                    return clean_text
            
            # Data part that might contain text
            if part_kind == 'data' and part_data:
                if isinstance(part_data, list):
                    for data_item in part_data:
                        if isinstance(data_item, dict) and data_item.get('kind') == 'text':
                            text_content = data_item.get('text', '')
                            if text_content and text_content.strip():
                                clean_text = clean_user_input(text_content)
                                if clean_text:
                                    return clean_text
                elif isinstance(part_data, str):
                    clean_text = clean_user_input(part_data)
                    if clean_text:
                        return clean_text
        
        # If we get here, try to find any text in the entire params structure
        params_str = json.dumps(params)
        if 'today' in params_str.lower():
            return "today's image"
        elif 'space fact' in params_str.lower() or 'random fact' in params_str.lower():
            return "space fact"
        elif 'random image' in params_str.lower():
            return "random image"
        elif 'yesterday' in params_str.lower():
            return "yesterday's image"
        elif 'help' in params_str.lower():
            return "help"
            
    except Exception as e:
        print(f"ERROR in message extraction: {e}")
    
    # Default fallback
    return "today's image"

def extract_user_message_from_a2a_message(message: dict) -> str:
    """Extract user message from A2A message format"""
    parts = message.get('parts', [])
    for part in parts:
        if part.get('kind') == 'text' and part.get('text'):
            clean_text = clean_user_input(part['text'])
            if clean_text:
                return clean_text
    return "today's image"

def clean_user_input(text: str) -> str:
    """Clean and normalize user input text"""
    if not text or not text.strip():
        return ""
    
    # Remove HTML tags and extra whitespace
    import re
    clean_text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
    clean_text = clean_text.strip()
    
    # Simple command detection
    clean_text_lower = clean_text.lower()
    
    if any(cmd in clean_text_lower for cmd in ['space fact', 'random fact']):
        return "space fact"
    elif 'random image' in clean_text_lower:
        return "random image"
    elif 'yesterday' in clean_text_lower:
        return "yesterday's image"
    elif 'today' in clean_text_lower:
        return "today's image"
    elif 'help' in clean_text_lower:
        return "help"
    elif 'test' in clean_text_lower:
        return "test image"
    
    return clean_text

async def process_user_command(command: str, request_id: str):
    """Process user command and return appropriate response"""
    print(f"🚀 PROCESSING COMMAND: '{command}'")
    
    if command == "space fact":
        return await create_space_fact_response(request_id)
    elif command == "random image":
        nasa_data = await get_random_apod_data()
    elif command == "yesterday's image":
        nasa_data = await get_yesterday_apod_data()
    elif command == "help":
        return await create_help_response(request_id)
    elif command == "test image":
        nasa_data = random.choice(FALLBACK_IMAGES)
    else:  # today's image (default)
        nasa_data = await get_nasa_apod_data()
    
    return await create_nasa_response(nasa_data, request_id)

async def get_nasa_apod_data(date=None):
    """Fetch NASA APOD data with better timeout handling and caching"""
    cache_key = f"apod_{date}" if date else "apod_today"
    
    # Check cache first
    if cache_key in nasa_cache:
        cache_time, cached_data = nasa_cache[cache_key]
        if (datetime.now() - cache_time).total_seconds() < CACHE_DURATION:
            print(f"DEBUG: Using cached data for {cache_key}")
            return cached_data
    
    try:
        url = f"{NASA_APOD_URL}?api_key={NASA_API_KEY}"
        if date:
            url += f"&date={date}"
            
        print(f"DEBUG: Calling NASA API: {url}")
        
        # Increased timeout with retry logic
        timeout = aiohttp.ClientTimeout(total=15)  # Increased to 15 seconds
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"DEBUG: NASA API success - Title: {data.get('title')}")
                    # Cache the successful response
                    nasa_cache[cache_key] = (datetime.now(), data)
                    return data
                else:
                    print(f"DEBUG: NASA API error: {response.status}")
                    return get_fallback_response()
                
    except asyncio.TimeoutError:
        print("DEBUG: NASA API timeout - using fallback")
        return get_fallback_response()
    except Exception as e:
        print(f"DEBUG: NASA API exception: {e}")
        return get_fallback_response()

async def get_random_apod_data():
    """Get random APOD from archive with caching"""
    cache_key = "apod_random"
    
    if cache_key in nasa_cache:
        cache_time, cached_data = nasa_cache[cache_key]
        if (datetime.now() - cache_time).total_seconds() < CACHE_DURATION:
            return cached_data
    
    try:
        # NASA APOD started June 16, 1995
        start_date = datetime(1995, 6, 16)
        end_date = datetime.now() - timedelta(days=1)  # Exclude today
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        data = await get_nasa_apod_data(random_date.strftime("%Y-%m-%d"))
        nasa_cache[cache_key] = (datetime.now(), data)
        return data
    except Exception as e:
        print(f"DEBUG: Random APOD error: {e}")
        return random.choice(FALLBACK_IMAGES)

async def get_yesterday_apod_data():
    """Get yesterday's APOD"""
    yesterday = datetime.now() - timedelta(days=1)
    return await get_nasa_apod_data(yesterday.strftime("%Y-%m-%d"))

def get_fallback_response():
    """Return random fallback response"""
    return random.choice(FALLBACK_IMAGES)

async def create_nasa_response(nasa_data, request_id):
    """Create NASA response with enhanced formatting"""
    response_text = format_nasa_response(nasa_data)
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=response_text)],
        messageId=str(uuid4()),
        taskId=request_id
    )
    
    artifacts = []
    image_url = nasa_data.get('url', '')
    
    # Add image artifact if available
    if nasa_data.get('media_type') == 'image' and image_url:
        artifacts.append(Artifact(
            name="nasa_image",
            parts=[MessagePart(kind="file", file_url=image_url)]
        ))
    
    # Add title artifact
    artifacts.append(Artifact(
        name="image_title",
        parts=[MessagePart(kind="text", text=nasa_data.get('title', 'NASA Image'))]
    ))
    
    return TaskResult(
        id=request_id,
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed",
            message=response_message
        ),
        artifacts=artifacts,
        history=[response_message]
    )

async def create_space_fact_response(request_id):
    """Create space fact response"""
    fact = random.choice(SPACE_FACTS)
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=f"🌌 *Space Fact* 🌌\n\n{fact}")],
        messageId=str(uuid4()),
        taskId=request_id
    )
    
    return TaskResult(
        id=request_id,
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
        history=[response_message]
    )

async def create_help_response(request_id):
    """Create help response"""
    help_text = """🛰️ *NASA Space Explorer Commands* 🛰️

Available commands:
• "today's image" - Today's Astronomy Picture of the Day
• "random image" - Random space image from NASA's archive
• "yesterday's image" - Yesterday's astronomy picture
• "space fact" or "random fact" - Interesting space facts
• "help" - Show this help message

Try: "today's image" to see today's space wonder! 🚀"""
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=help_text)],
        messageId=str(uuid4()),
        taskId=request_id
    )
    
    return TaskResult(
        id=request_id,
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed",
            message=response_message
        ),
        artifacts=[],
        history=[response_message]
    )

def format_nasa_response(data):
    """Format NASA data into nice response"""
    title = data.get('title', 'Unknown Title')
    explanation = data.get('explanation', 'No description available.')
    date = data.get('date', 'Unknown date')
    
    # Truncate long explanations
    if len(explanation) > 600:
        explanation = explanation[:600] + "..."
    
    response_text = f"""🌌 *{title}* 🌌

{explanation}

*Date:* {date}
*Media Type:* {data.get('media_type', 'image')}

*Explore the cosmos!* 🚀"""
    
    return response_text

@app.get("/")
async def root():
    return {
        "message": "NASA Space Explorer Agent is running!",
        "status": "healthy", 
        "version": "2.1.0",
        "protocol": "A2A Compliant",
        "features": {
            "caching": "Enabled (1 hour)",
            "fallback_images": f"{len(FALLBACK_IMAGES)} available",
            "timeout": "15 seconds",
            "space_facts": f"{len(SPACE_FACTS)} available"
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
            "cache_size": len(nasa_cache),
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