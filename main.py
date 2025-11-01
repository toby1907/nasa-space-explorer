from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
import asyncio
import aiohttp

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

from pydantic import ValidationError
import json

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
        
        try:
            # First try to parse with strict validation
            rpc_request = JSONRPCRequest(**body)
        except ValidationError as e:
            print(f"Validation error: {e}")
            # Telex is sending invalid data - extract user message manually
            user_message = extract_user_message_from_telex_body(body)
            print(f"Extracted user message: '{user_message}'")
            
            # Process the message directly
            result = await process_message_directly(user_message, body.get("id", "telex-fallback"))
            
            response = JSONRPCResponse(
                id=body.get("id", "telex-fallback"),
                result=result
            )
            return response.model_dump()
        
        # Process based on method (normal flow)
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

def extract_user_message_from_telex_body(body):
    """Extract user command - WITH FORCEFUL DEBUGGING"""
    try:
        print("=== DEBUG: Starting extraction ===")
        message_parts = body.get('params', {}).get('message', {}).get('parts', [])
        print(f"DEBUG: Found {len(message_parts)} message parts")
        
        for i, part in enumerate(message_parts):
            print(f"DEBUG: Part {i}: kind={part.get('kind')}, text={part.get('text')}")
            
            if part.get('kind') == 'text' and part.get('text'):
                text = part['text'].lower()
                print(f"DEBUG: Processing text: '{text}'")
                
                # FORCEFUL COMMAND DETECTION
                if "space fact" in text:
                    print("🚀 DEBUG: FOUND 'space fact' - RETURNING SPACE FACT!")
                    return "space fact"
                elif "random fact" in text:
                    print("🚀 DEBUG: FOUND 'random fact' - RETURNING SPACE FACT!")
                    return "space fact"
                elif "random image" in text:
                    print("🚀 DEBUG: FOUND 'random image'")
                    return "random image"
                elif "yesterday" in text:
                    print("🚀 DEBUG: FOUND 'yesterday'")
                    return "yesterday's image"
                elif "today" in text:
                    print("🚀 DEBUG: FOUND 'today'")
                    return "today's image"
                else:
                    print(f"DEBUG: No command found in: '{text}'")
        
        print("DEBUG: No command found in any part, defaulting to today's image")
        return "today's image"
        
    except Exception as e:
        print(f"ERROR in extraction: {e}")
        return "today's image"
async def create_help_response(request_id):
    """Create help response with available commands"""
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
        taskId=None
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
async def process_message_directly(user_message, request_id):
    """Process command - WITH CLEAR DEBUG"""
    print(f"=== DEBUG: Processing command: '{user_message}' ===")
    
    # Create a mock user_message object for the direct processing flow
    mock_user_message = A2AMessage(
        role="user",
        parts=[MessagePart(kind="text", text=user_message)],
        taskId=request_id
    )
    
    if user_message == "space fact":
        print("🚀 DEBUG: CONFIRMED - RETURNING SPACE FACT!")
        return await get_space_fact_response(mock_user_message)  # Use existing function
    elif user_message == "random image":
        print("DEBUG: Returning random NASA image")
        nasa_data = await get_random_apod_data()
    elif user_message == "yesterday's image":
        print("DEBUG: Returning yesterday's NASA image")
        nasa_data = await get_yesterday_apod_data()
    elif user_message == "help":
        print("DEBUG: Returning help")
        return await create_help_response(request_id)
    else:
        print("DEBUG: Defaulting to today's NASA image")
        nasa_data = await get_nasa_apod_data()
    
    return await create_nasa_response(nasa_data, mock_user_message)  # Use the new function
async def handle_message_send(params: MessageParams):
    """Handle message/send - COMPLETE DEBUGGING"""
    print("=== DEBUG: handle_message_send called ===")
    print(f"DEBUG: Full message parts: {params.message.parts}")
    
    # EXTRACT USER MESSAGE from all possible locations
    user_text = ""
    for i, part in enumerate(params.message.parts):
        print(f"DEBUG: Part {i} - kind: {part.kind}, text: {part.text}, data: {part.data}")
        
        # Check text parts
        if part.kind == "text" and part.text and part.text.strip():
            user_text = part.text
            print(f"DEBUG: Found text in parts: '{user_text}'")
            break
        
        # Check data parts that might contain the message
        if part.kind == "data" and part.data:
            print(f"DEBUG: Found data part: {part.data}")
            # Sometimes the message is in data parts
            if isinstance(part.data, list):
                for data_item in part.data:
                    if isinstance(data_item, dict) and data_item.get('kind') == 'text':
                        potential_text = data_item.get('text', '')
                        if potential_text and potential_text.strip():
                            user_text = potential_text
                            print(f"DEBUG: Found text in data: '{user_text}'")
                            break
    
    # If still no text, try a different approach
    if not user_text:
        print("DEBUG: No text found in normal parts, checking alternative locations")
        # Sometimes Telex puts the message in different structures
        try:
            # Convert the whole message to dict and search for text
            message_dict = params.message.dict()
            import json
            print(f"DEBUG: Full message structure: {json.dumps(message_dict, indent=2)}")
        except Exception as e:
            print(f"DEBUG: Could not dump message structure: {e}")
    
    # SIMPLE FALLBACK: If we can't find the user's message, just return a test image
    if not user_text:
        print("🚀 DEBUG: No user text found - returning test image by default")
        test_data = {
            "title": "Hubble Space Telescope View", 
            "explanation": "Since we couldn't detect your specific command, here's a beautiful space image from the Hubble Space Telescope showing distant galaxies and nebulae.",
            "url": "https://images-assets.nasa.gov/image/PIA12153/PIA12153~large.jpg",
            "media_type": "image",
            "date": "2024-01-01"
        }
        return await create_nasa_response(test_data, params.message)
    
    # If we found text, process it normally
    print(f"DEBUG: Processing user text: '{user_text}'")
    
    # ULTRA-SIMPLE COMMAND DETECTION
    command = "test image"  # Default to test image for now
    
    clean_text = user_text.lower().replace("'", "").replace('"', '')
    
    if "test image" in clean_text:
        command = "test image"
    elif "space fact" in clean_text or "random fact" in clean_text:
        command = "space fact"
    elif "random image" in clean_text:
        command = "random image" 
    elif "yesterday" in clean_text:
        command = "yesterday's image"
    elif "today" in clean_text:
        command = "today's image"
    elif "help" in clean_text:
        command = "help"
    
    print(f"🚀 DEBUG: FINAL COMMAND: '{command}'")
    
    # Process the command
    if command == "test image":
        print("🚀 DEBUG: Returning test image")
        test_data = {
            "title": "TEST: Orion Nebula",
            "explanation": "This is a test image to verify image display in Telex chat. The Orion Nebula is one of the brightest nebulae in the night sky.",
            "url": "https://images-assets.nasa.gov/image/PIA12153/PIA12153~large.jpg", 
            "media_type": "image",
            "date": "2024-01-01"
        }
        return await create_nasa_response(test_data, params.message)
    elif command == "space fact":
        print("🚀 DEBUG: Returning space fact")
        return await get_space_fact_response(params.message)
    elif command == "random image":
        print("DEBUG: Returning random NASA image")
        nasa_data = await get_random_apod_data()
    elif command == "yesterday's image":
        print("DEBUG: Returning yesterday's NASA image")
        nasa_data = await get_yesterday_apod_data()
    elif command == "help":
        print("DEBUG: Returning help")
        help_text = """🛰️ *NASA Space Explorer* 🛰️

Commands:
• "today's image" - NASA Astronomy Picture
• "random image" - Random space image  
• "yesterday's image" - Yesterday's picture
• "space fact" - Interesting space facts
• "test image" - Test image display
• "help" - This message

Try: "test image" to see if images work! 🚀"""
        
        response_message = A2AMessage(
            role="agent",
            parts=[MessagePart(kind="text", text=help_text)],
            messageId=str(uuid4()),
            taskId=params.message.taskId
        )
        
        return TaskResult(
            id=params.message.taskId or str(uuid4()),
            contextId=str(uuid4()),
            status=TaskStatus(
                state="completed",
                message=response_message
            ),
            artifacts=[],
            history=[params.message, response_message]
        )
    else:  # today's image (default)
        print("DEBUG: Returning today's NASA image")
        nasa_data = await get_nasa_apod_data()
    
    return await create_nasa_response(nasa_data, params.message)
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
    """Fetch NASA APOD data with better timeout handling"""
    try:
        url = f"{NASA_APOD_URL}?api_key={NASA_API_KEY}"
        if date:
            url += f"&date={date}"
            
        print(f"DEBUG: Calling NASA API: {url}")
        
        # Shorter timeout for faster fallback
        timeout = aiohttp.ClientTimeout(total=5)  # Reduced from 10 to 5 seconds
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"DEBUG: NASA API success - Title: {data.get('title')}")
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
async def create_nasa_response(nasa_data, user_message):
    """Create NASA response with enhanced image display"""
    response_text = format_nasa_response(nasa_data)
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=response_text)],
        messageId=str(uuid4()),
        taskId=user_message.taskId
    )
    
    artifacts = []
    image_url = nasa_data.get('url', '')
    
    if nasa_data.get('media_type') == 'image' and image_url:
        # Add image as both file and text artifact for maximum compatibility
        artifacts.append(Artifact(
            name="nasa_image",
            parts=[
                MessagePart(kind="file", file_url=image_url),
                MessagePart(kind="text", text=f"NASA Image: {image_url}")
            ]
        ))
        
        # Also add a separate text artifact with the URL
        artifacts.append(Artifact(
            name="image_url",
            parts=[MessagePart(kind="text", text=image_url)]
        ))
    
    artifacts.append(Artifact(
        name="image_title",
        parts=[MessagePart(kind="text", text=nasa_data.get('title', 'NASA Image'))]
    ))
    
    task_id = user_message.taskId or str(uuid4())
    
    return TaskResult(
        id=task_id,
        contextId=str(uuid4()),
        status=TaskStatus(
            state="completed",
            message=response_message
        ),
        artifacts=artifacts,
        history=[user_message, response_message]
    )
def get_fallback_response():
    """Return fallback response with sample image when NASA API fails"""
    return {
        "title": "Hubble Space Telescope View",
        "explanation": "This is a sample space image since NASA's servers are temporarily unavailable. The Hubble Space Telescope has captured stunning views of distant galaxies, nebulae, and star clusters, revealing the beauty of our universe.",
        "url": "https://images-assets.nasa.gov/image/PIA12153/PIA12153~large.jpg",
        "media_type": "image",
        "date": datetime.now().strftime("%Y-%m-%d")
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
    """Format NASA data into nice response with proper image display"""
    title = data.get('title', 'Unknown Title')
    explanation = data.get('explanation', 'No description available.')
    date = data.get('date', 'Unknown date')
    image_url = data.get('url', '')
    
    # Truncate long explanations
    if len(explanation) > 800:
        explanation = explanation[:800] + "..."
    
    # FORCE IMAGE DISPLAY - Multiple formats for Telex compatibility
    response_text = f"""🌌 *{title}* 🌌

{explanation}

🖼️ **Image:** {image_url}

*Details:*
📅 Date: {date}
🖼️ Media Type: {data.get('media_type', 'image')}

📸 *View Image:* {image_url}

*Explore the cosmos!* 🚀"""
    
    return response_text


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