from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from uuid import uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
import asyncio
import aiohttp
import random
import json
import re

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
    metadata: Optional[Dict[str, Any]] = None

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
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    message: Optional[A2AMessage] = None

class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: List[MessagePart]

class TaskResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    contextId: str = Field(default_factory=lambda: str(uuid4()))
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
    version="4.0.0"
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
CACHE_DURATION = 3600  # 1 hour cache

# Pre-defined fallback images
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
    }
]

SPACE_FACTS = [
    "A day on Mercury lasts 59 Earth days!",
    "Neptune's winds can reach 1,600 km/h - the fastest in the solar system!",
    "There are more stars in the universe than grains of sand on all Earth's beaches!",
    "A teaspoon of neutron star would weigh about 6 billion tons!",
    "Venus is the only planet that spins clockwise!",
]

@app.post("/a2a/nasa")
async def a2a_endpoint(request: Request):
    """A2A endpoint that matches the working agent format exactly"""
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
        
        # Extract user message from the request
        user_message = extract_user_message_from_request(body)
        print(f"🚀 DETECTED COMMAND: '{user_message}'")
        
        # Process the command and get response
        result = await process_user_command(user_message, request_id, body)
        
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

def extract_user_message_from_request(body: dict) -> str:
    """Extract user message from request body - SIMPLIFIED"""
    try:
        params = body.get('params', {})
        message = params.get('message', {})
        parts = message.get('parts', [])
        
        # Look for the most recent user message in data parts
        for part in parts:
            if part.get('kind') == 'data' and part.get('data'):
                data_items = part['data']
                if isinstance(data_items, list):
                    # Look for the last user message (most recent)
                    for item in reversed(data_items):
                        if (isinstance(item, dict) and 
                            item.get('kind') == 'text' and 
                            item.get('text') and
                            not is_bot_response(item.get('text', ''))):
                            text = item['text'].strip()
                            if text and text.startswith('<p>') and text.endswith('</p>'):
                                # Extract clean command from HTML
                                clean_text = re.sub('<[^<]+?>', '', text).strip()
                                command = detect_command(clean_text)
                                if command:
                                    return command
        
        return "today's image"
            
    except Exception as e:
        print(f"Error extracting message: {e}")
        return "today's image"

def is_bot_response(text: str) -> bool:
    """Check if text is a bot response"""
    bot_indicators = ['fetching', 'here\'s', 'view image', 'click the link', 'astronomy picture']
    return any(indicator in text.lower() for indicator in bot_indicators)

def detect_command(text: str) -> str:
    """Detect command from clean text"""
    clean_text = text.lower().strip()
    
    if any(cmd in clean_text for cmd in ['space fact', 'random fact']):
        return "space fact"
    elif 'random image' in clean_text:
        return "random image"
    elif 'yesterday' in clean_text:
        return "yesterday's image"
    elif 'today' in clean_text:
        return "today's image"
    elif 'help' in clean_text:
        return "help"
    elif 'test' in clean_text:
        return "test image"
    
    return ""

async def process_user_command(command: str, request_id: str, original_body: dict):
    """Process user command and return response in correct format"""
    print(f"🚀 PROCESSING: '{command}'")
    
    if command == "space fact":
        response_data = await create_space_fact_data()
    elif command == "random image":
        response_data = await get_random_apod_data()
    elif command == "yesterday's image":
        response_data = await get_yesterday_apod_data()
    elif command == "help":
        response_data = create_help_data()
    elif command == "test image":
        response_data = random.choice(FALLBACK_IMAGES)
    else:  # today's image (default)
        response_data = await get_nasa_apod_data()
    
    return await create_response_in_correct_format(response_data, request_id, original_body)

async def get_nasa_apod_data(date=None):
    """Fetch NASA APOD data"""
    cache_key = f"apod_{date}" if date else "apod_today"
    
    if cache_key in nasa_cache:
        cache_time, cached_data = nasa_cache[cache_key]
        if (datetime.now() - cache_time).total_seconds() < CACHE_DURATION:
            return cached_data
    
    try:
        url = f"{NASA_APOD_URL}?api_key={NASA_API_KEY}"
        if date:
            url += f"&date={date}"
        
        timeout = aiohttp.ClientTimeout(total=8)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    nasa_cache[cache_key] = (datetime.now(), data)
                    return data
                else:
                    return get_fallback_response()
                
    except Exception:
        return get_fallback_response()

async def get_random_apod_data():
    """Get random APOD from archive"""
    try:
        start_date = datetime(1995, 6, 16)
        end_date = datetime.now() - timedelta(days=1)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        return await get_nasa_apod_data(random_date.strftime("%Y-%m-%d"))
    except Exception:
        return random.choice(FALLBACK_IMAGES)

async def get_yesterday_apod_data():
    """Get yesterday's APOD"""
    yesterday = datetime.now() - timedelta(days=1)
    return await get_nasa_apod_data(yesterday.strftime("%Y-%m-%d"))

def get_fallback_response():
    """Return random fallback response"""
    return random.choice(FALLBACK_IMAGES)

async def create_space_fact_data():
    """Create space fact data"""
    return {
        "type": "space_fact",
        "content": random.choice(SPACE_FACTS),
        "title": "Space Fact"
    }

def create_help_data():
    """Create help data"""
    return {
        "type": "help",
        "content": """🛰️ *NASA Space Explorer Commands* 🛰️

**Available Commands:**
• "today's image" - Today's Astronomy Picture of the Day  
• "random image" - Random space image from NASA's archive
• "yesterday's image" - Yesterday's astronomy picture
• "space fact" or "random fact" - Interesting space facts
• "help" - Show this help message

**Try:** "today's image" to see today's space wonder! 🚀"""
    }

async def create_response_in_correct_format(response_data, request_id: str, original_body: dict):
    """Create response in the EXACT format that works with Telex"""
    
    # Generate response text based on data type
    if response_data.get('type') == 'space_fact':
        response_text = f"🌌 *Space Fact* 🌌\n\n{response_data['content']}\n\n*Learn something new every day!* 🚀"
    elif response_data.get('type') == 'help':
        response_text = response_data['content']
    else:
        # NASA image response
        title = response_data.get('title', 'Unknown Title')
        explanation = response_data.get('explanation', 'No description available.')
        date = response_data.get('date', 'Unknown date')
        image_url = response_data.get('url', '')
        
        if len(explanation) > 500:
            explanation = explanation[:500] + "..."
        
        response_text = f"""🛰️ **{title}** 🛰️

{explanation}

**Date:** {date}
**Type:** {response_data.get('media_type', 'image')}"""

        if response_data.get('media_type') == 'image' and image_url:
            response_text += f"\n\n📸 **View Image:** {image_url}"
            response_text += f"\n\n🔗 **Click the link above to view the image!**"
        
        response_text += "\n\n*Explore the cosmos!* 🚀"

    # Create the main response message (EXACT format from working agent)
    response_message = A2AMessage(
        role="agent",
        parts=[
            MessagePart(
                kind="text",
                text=response_text,
                data=None,
                file_url=None
            )
        ],
        messageId=str(uuid4()),
        taskId=str(uuid4()),  # Different from request_id like in working example
        metadata=None
    )

    # Create artifacts with the SAME text as message (like working example)
    artifacts = [
        Artifact(
            artifactId=str(uuid4()),
            name="nasa_response",
            parts=[
                MessagePart(
                    kind="text",
                    text=response_text,  # SAME text as in message
                    data=None,
                    file_url=None
                )
            ]
        )
    ]

    # Build history from the original request (like working example does)
    history = await build_conversation_history(original_body, response_message)

    # Create task result with EXACT same structure as working example
    result = TaskResult(
        id=str(uuid4()),  # Different from request_id like in working example
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed",
            timestamp=datetime.utcnow().isoformat() + "Z",  # Note the Z at end
            message=response_message
        ),
        artifacts=artifacts,
        history=history,
        kind="task"
    )

    return result

async def build_conversation_history(original_body: dict, current_response: A2AMessage) -> List[A2AMessage]:
    """Build conversation history from original request"""
    history = []
    
    try:
        params = original_body.get('params', {})
        original_message = params.get('message', {})
        original_parts = original_message.get('parts', [])
        
        # Extract previous messages from data parts
        for part in original_parts:
            if part.get('kind') == 'data' and part.get('data'):
                data_items = part['data']
                if isinstance(data_items, list):
                    for item in data_items:
                        if isinstance(item, dict) and item.get('kind') == 'text' and item.get('text'):
                            text = item['text']
                            # Determine role based on content
                            if text.startswith('<p>') and text.endswith('</p>'):
                                # User message
                                clean_text = re.sub('<[^<]+?>', '', text).strip()
                                history.append(A2AMessage(
                                    role="user",
                                    parts=[MessagePart(kind="text", text=clean_text)],
                                    messageId=str(uuid4()),
                                    taskId=None,
                                    metadata=None
                                ))
                            else:
                                # Assume agent message for non-HTML text
                                history.append(A2AMessage(
                                    role="agent", 
                                    parts=[MessagePart(kind="text", text=text)],
                                    messageId=str(uuid4()),
                                    taskId=None,
                                    metadata=None
                                ))
        
        # Add current response to history
        history.append(current_response)
        
    except Exception as e:
        print(f"Error building history: {e}")
        # Fallback: just add current response
        history = [current_response]
    
    return history

@app.get("/")
async def root():
    return {
        "message": "NASA Space Explorer Agent - UPDATED FORMAT",
        "status": "healthy", 
        "version": "4.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NASA Space Explorer Agent"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)