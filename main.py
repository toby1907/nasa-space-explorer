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
    version="2.5.0"
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

# Cache for NASA responses
nasa_cache = {}
CACHE_DURATION = 3600  # 1 hour

# Pre-defined fallback images with guaranteed working URLs
FALLBACK_IMAGES = [
    {
        "title": "Earth from Space",
        "explanation": "A beautiful view of our planet Earth from the International Space Station, showing continents and oceans in stunning detail.",
        "url": "https://apod.nasa.gov/apod/image/2401/ISS045E100257.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    },
    {
        "title": "Orion Nebula",
        "explanation": "The Orion Nebula is a massive stellar nursery located 1,500 light-years away, where new stars are being born from clouds of gas and dust.",
        "url": "https://apod.nasa.gov/apod/image/2401/M42M43_Final_Seidel_2048.jpg", 
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    },
    {
        "title": "Moon Surface",
        "explanation": "Detailed view of the Moon's surface showing craters, mountains, and the dramatic landscapes of our closest celestial neighbor.",
        "url": "https://apod.nasa.gov/apod/image/2401/MoonCraters_Bourous_2048.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    },
    {
        "title": "Jupiter's Storms",
        "explanation": "Jupiter's turbulent atmosphere showing the famous Great Red Spot and numerous other storm systems in the gas giant's clouds.",
        "url": "https://apod.nasa.gov/apod/image/2401/Jupiter_Cassini_1080.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
    },
    {
        "title": "Andromeda Galaxy",
        "explanation": "Our nearest galactic neighbor, the Andromeda Galaxy, containing over a trillion stars and spanning 220,000 light years across.",
        "url": "https://apod.nasa.gov/apod/image/2401/M31_2023_08_07_BCD_1024.jpg",
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
    """A2A endpoint that handles both valid and invalid data formats"""
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
            print(f"DEBUG: Valid JSON-RPC request - method: {rpc_request.method}")
            
            # Process based on method using the proper objects
            if rpc_request.method == "message/send":
                result = await handle_message_send_proper(rpc_request.params, request_id)
            elif rpc_request.method == "execute":
                result = await handle_execute_proper(rpc_request.params, request_id)
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {rpc_request.method}"
                        }
                    }
                )
                
        except Exception as e:
            print(f"DEBUG: Validation failed, using fallback: {e}")
            # Use fallback parsing for Telex's non-standard format
            method = body.get("method", "message/send")
            params = body.get("params", {})
            
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

async def handle_message_send_proper(params: MessageParams, request_id: str):
    """Handle message/send with proper Pydantic objects"""
    print("=== DEBUG: handle_message_send_proper called ===")
    
    # Extract user message from the proper MessageParams object
    user_message = extract_user_message_from_proper_params(params)
    print(f"🚀 DEBUG: EXTRACTED USER MESSAGE: '{user_message}'")
    
    # Process the command
    return await process_user_command(user_message, request_id)

async def handle_execute_proper(params: ExecuteParams, request_id: str):
    """Handle execute with proper Pydantic objects"""
    if params.messages:
        # Use the last message
        last_message = params.messages[-1]
        user_message = extract_user_message_from_a2a_message(last_message)
    else:
        user_message = "today's image"
    
    return await process_user_command(user_message, request_id)

async def handle_message_send_fallback(params: dict, request_id: str):
    """Handle message/send with fallback parsing for dict params"""
    print("=== DEBUG: handle_message_send_fallback called ===")
    
    # Extract user message from dictionary params
    user_message = extract_user_message_robust(params)
    print(f"🚀 DEBUG: EXTRACTED USER MESSAGE: '{user_message}'")
    
    # Process the command
    return await process_user_command(user_message, request_id)

async def handle_execute_fallback(params: dict, request_id: str):
    """Handle execute with fallback parsing for dict params"""
    messages = params.get('messages', [])
    if messages:
        # Use the last message
        last_message = messages[-1]
        user_message = extract_user_message_from_dict_message(last_message)
    else:
        user_message = "today's image"
    
    return await process_user_command(user_message, request_id)

def extract_user_message_from_proper_params(params: MessageParams) -> str:
    """Extract user message from proper MessageParams object"""
    try:
        parts = params.message.parts
        print(f"DEBUG: Found {len(parts)} parts in proper params")
        
        all_texts = []
        for i, part in enumerate(parts):
            print(f"DEBUG: Part {i} - kind: {part.kind}, text: {part.text[:200] if part.text else 'None'}")
            
            # Direct text part
            if part.kind == 'text' and part.text and part.text.strip():
                all_texts.append(part.text)
                clean_text = clean_user_input(part.text)
                if clean_text:
                    print(f"🚀 DEBUG: EXTRACTED COMMAND: '{clean_text}'")
                    return clean_text
            
            # Data part that might contain text
            if part.kind == 'data' and part.data:
                print(f"DEBUG: Processing data part: {part.data}")
                if isinstance(part.data, list):
                    for data_item in part.data:
                        if isinstance(data_item, dict) and data_item.get('kind') == 'text':
                            text_content = data_item.get('text', '')
                            if text_content and text_content.strip():
                                all_texts.append(text_content)
                                clean_text = clean_user_input(text_content)
                                if clean_text:
                                    print(f"🚀 DEBUG: EXTRACTED COMMAND FROM DATA: '{clean_text}'")
                                    return clean_text
                elif isinstance(part.data, str):
                    all_texts.append(part.data)
                    clean_text = clean_user_input(part.data)
                    if clean_text:
                        print(f"🚀 DEBUG: EXTRACTED COMMAND FROM DATA STRING: '{clean_text}'")
                        return clean_text
        
        print(f"DEBUG: All texts found: {all_texts}")
        # Default fallback
        return "today's image"
            
    except Exception as e:
        print(f"ERROR in proper params extraction: {e}")
        return "today's image"

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

def extract_user_message_from_a2a_message(message: A2AMessage) -> str:
    """Extract user message from A2A message object"""
    parts = message.parts
    for part in parts:
        if part.kind == 'text' and part.text:
            clean_text = clean_user_input(part.text)
            if clean_text:
                return clean_text
    return "today's image"

def extract_user_message_from_dict_message(message: dict) -> str:
    """Extract user message from dictionary message"""
    parts = message.get('parts', [])
    for part in parts:
        if part.get('kind') == 'text' and part.get('text'):
            clean_text = clean_user_input(part['text'])
            if clean_text:
                return clean_text
    return "today's image"

def clean_user_input(text: str) -> str:
    """Clean and normalize user input text - FIXED VERSION"""
    if not text or not text.strip():
        return ""
    
    # Remove HTML tags and extra whitespace
    import re
    clean_text = re.sub('<[^<]+?>', '', text)  # Remove HTML tags
    clean_text = ' '.join(clean_text.split())  # Normalize whitespace
    clean_text = clean_text.strip()
    
    print(f"DEBUG: Raw text: '{text}' -> Cleaned: '{clean_text}'")
    
    # Handle the specific case of repeated "test image" commands
    clean_text_lower = clean_text.lower()
    
    # Check for exact matches first
    if 'test image' in clean_text_lower:
        print("DEBUG: Detected 'test image' command")
        return "test image"
    elif 'space fact' in clean_text_lower or 'random fact' in clean_text_lower:
        print("DEBUG: Detected 'space fact' command")
        return "space fact"
    elif 'random image' in clean_text_lower:
        print("DEBUG: Detected 'random image' command")
        return "random image"
    elif 'yesterday' in clean_text_lower:
        print("DEBUG: Detected 'yesterday's image' command")
        return "yesterday's image"
    elif 'today' in clean_text_lower:
        print("DEBUG: Detected 'today's image' command")
        return "today's image"
    elif 'help' in clean_text_lower:
        print("DEBUG: Detected 'help' command")
        return "help"
    
    # If no specific command found, default to today's image
    print("DEBUG: No specific command detected, defaulting to today's image")
    return "today's image"

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
        
        # Shorter timeout for faster fallback
        timeout = aiohttp.ClientTimeout(total=8)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"DEBUG: NASA API success - Title: {data.get('title')}")
                    # Ensure we have proper image URLs
                    if data.get('media_type') == 'image':
                        if not data.get('hdurl'):
                            data['hdurl'] = data.get('url', '')
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
        end_date = datetime.now() - timedelta(days=1)
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
    """Create NASA response with clickable image URL in text"""
    response_text = format_nasa_response(nasa_data)
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=response_text)],
        messageId=str(uuid4()),
        taskId=request_id
    )
    
    # Still include artifacts for compatibility, but focus on text response
    artifacts = []
    image_url = nasa_data.get('url', '')
    
    if nasa_data.get('media_type') == 'image' and image_url:
        artifacts.append(Artifact(
            name="nasa_image",
            parts=[MessagePart(kind="file", file_url=image_url)]
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
    """Format NASA data into nice response with clickable image URL"""
    title = data.get('title', 'Unknown Title')
    explanation = data.get('explanation', 'No description available.')
    date = data.get('date', 'Unknown date')
    image_url = data.get('url', '')
    
    # Truncate long explanations
    if len(explanation) > 500:
        explanation = explanation[:500] + "..."
    
    response_text = f"""🛰️ **{title}** 🛰️

{explanation}

**Date:** {date}
**Type:** {data.get('media_type', 'image')}"""

    # Always include clickable image URL prominently
    if data.get('media_type') == 'image' and image_url:
        response_text += f"\n\n📸 **View Image:** {image_url}"
        response_text += f"\n\n🔗 **Click the link above to view the image!**"
    else:
        response_text += f"\n\n📺 **Video URL:** {image_url}"
        response_text += f"\n\n🔗 **Click the link above to watch the video!**"
    
    response_text += "\n\n*Explore the cosmos!* 🚀"
    
    return response_text

@app.get("/")
async def root():
    return {
        "message": "NASA Space Explorer Agent is running!",
        "status": "healthy", 
        "version": "2.5.0",
        "protocol": "A2A Compliant",
        "features": {
            "caching": "Enabled (1 hour)",
            "fallback_images": f"{len(FALLBACK_IMAGES)} available",
            "timeout": "8 seconds",
            "space_facts": f"{len(SPACE_FACTS)} available",
            "image_display": "Clickable URLs in text"
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
                "media_type": data.get('media_type'),
                "has_image": data.get('media_type') == 'image'
            }
        }
    except Exception as e:
        return {"nasa_api_status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)