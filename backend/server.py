from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import base64
import aiofiles
import tempfile

# OpenAI for Whisper
from openai import OpenAI

# Google Cloud TTS
from google.cloud import texttospeech

# Emergent integrations for Gemini
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize AI clients (lazily to avoid startup errors)
openai_client = None
def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=os.environ.get('EMERGENT_LLM_KEY'))
    return openai_client

# Create the main app without a prefix
app = FastAPI(title="Menene - Hausa Conversational AI")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== MODELS ====================

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: str  # 'user' or 'assistant'
    content: str
    audio_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Conversation"
    language: str = "ha"  # Hausa by default
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TranscriptionRequest(BaseModel):
    user_id: str
    conversation_id: str


class ChatRequest(BaseModel):
    conversation_id: str
    user_id: str
    message: str
    language: str = "ha"


class TTSRequest(BaseModel):
    text: str
    language: str = "ha-NG"
    voice: str = "ha-NG-Standard-A"


class ConversationCreate(BaseModel):
    user_id: str
    language: str = "ha"


# ==================== SPEECH TO TEXT ENDPOINT ====================

@api_router.post("/speech-to-text")
async def transcribe_audio(
    audio: UploadFile = File(...),
    user_id: str = File(...),
    conversation_id: str = File(...)
):
    """
    Transcribe Hausa audio using OpenAI Whisper API
    """
    try:
        logger.info(f"Received audio file: {audio.filename}, size: {audio.size}")
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a")
        temp_path = temp_file.name
        
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        # Transcribe using Whisper
        with open(temp_path, 'rb') as audio_file:
            transcription = get_openai_client().audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ha",  # Hausa language
                response_format="verbose_json"
            )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        transcribed_text = transcription.text
        
        # Save message to database
        message = Message(
            conversation_id=conversation_id,
            role="user",
            content=transcribed_text
        )
        
        await db.messages.insert_one(message.dict())
        
        # Update conversation timestamp
        await db.conversations.update_one(
            {"id": conversation_id},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        logger.info(f"Transcription successful: {transcribed_text[:50]}...")
        
        return {
            "success": True,
            "transcription": transcribed_text,
            "message_id": message.id,
            "duration": transcription.duration if hasattr(transcription, 'duration') else None
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ==================== CHAT ENDPOINT ====================

@api_router.post("/chat")
async def chat(request: ChatRequest):
    """
    Generate conversational AI response using Google Gemini
    """
    try:
        logger.info(f"Chat request for conversation: {request.conversation_id}")
        
        # Get conversation history
        messages = await db.messages.find(
            {"conversation_id": request.conversation_id}
        ).sort("timestamp", 1).to_list(100)
        
        # Initialize Gemini chat
        system_message = f"""You are Menene, a helpful AI assistant that speaks {request.language if request.language == 'ha' else 'Hausa'} language. 
You are friendly, knowledgeable, and culturally aware of West African contexts, particularly Nigeria. 
Respond naturally in Hausa language. Keep responses concise and helpful."""
        
        chat_session = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=request.conversation_id,
            system_message=system_message
        )
        
        # Configure to use Gemini
        chat_session.with_model("gemini", "gemini-3-flash-preview")
        
        # Create user message
        user_message = UserMessage(text=request.message)
        
        # Get response from Gemini
        response = await chat_session.send_message(user_message)
        
        # Save assistant message to database
        assistant_message = Message(
            conversation_id=request.conversation_id,
            role="assistant",
            content=response
        )
        
        await db.messages.insert_one(assistant_message.dict())
        
        # Update conversation
        await db.conversations.update_one(
            {"id": request.conversation_id},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        logger.info(f"Chat response generated: {response[:50]}...")
        
        return {
            "success": True,
            "response": response,
            "message_id": assistant_message.id
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ==================== TEXT TO SPEECH ENDPOINT ====================

@api_router.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using Google Cloud TTS (English fallback as Hausa is not supported)
    """
    try:
        logger.info(f"TTS request for text: {request.text[:50]}...")
        
        # Use English voice as Hausa voices are not supported by Google Cloud TTS
        language = "en-US"
        voice = "en-US-Standard-A"
        
        # Check cache first
        cached_audio = await db.audio_cache.find_one({
            "text": request.text,
            "language": language,
            "voice": voice
        })
        
        if cached_audio:
            logger.info("Returning cached audio")
            return {
                "success": True,
                "audio_content": cached_audio["audio_content"],
                "cached": True
            }
        
        # Use Google Cloud TTS API directly with API key
        import requests
        
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={os.environ.get('GOOGLE_TTS_API_KEY')}"
        
        payload = {
            "input": {"text": request.text},
            "voice": {
                "languageCode": language,
                "name": voice
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "pitch": 0,
                "speakingRate": 1.0
            }
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"TTS API error: {response.text}"
            )
        
        audio_content = response.json().get("audioContent")
        
        # Cache the audio
        await db.audio_cache.insert_one({
            "text": request.text,
            "language": language,
            "voice": voice,
            "audio_content": audio_content,
            "created_at": datetime.utcnow()
        })
        
        logger.info("TTS audio generated and cached")
        
        return {
            "success": True,
            "audio_content": audio_content,
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


# ==================== CONVERSATION MANAGEMENT ====================

@api_router.post("/conversations", response_model=Conversation)
async def create_conversation(input: ConversationCreate):
    """Create a new conversation"""
    conversation = Conversation(
        user_id=input.user_id,
        language=input.language
    )
    await db.conversations.insert_one(conversation.dict())
    return conversation


@api_router.get("/conversations/{user_id}")
async def get_conversations(user_id: str):
    """Get all conversations for a user"""
    conversations = await db.conversations.find(
        {"user_id": user_id}, {"_id": 0}  # Exclude _id field to avoid ObjectId serialization issues
    ).sort("updated_at", -1).to_list(100)
    
    return {"success": True, "conversations": conversations}


@api_router.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    """Get all messages in a conversation"""
    messages = await db.messages.find(
        {"conversation_id": conversation_id}, {"_id": 0}  # Exclude _id field to avoid ObjectId serialization issues
    ).sort("timestamp", 1).to_list(1000)
    
    return {"success": True, "messages": messages}


@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its messages"""
    await db.conversations.delete_one({"id": conversation_id})
    await db.messages.delete_many({"conversation_id": conversation_id})
    
    return {"success": True, "message": "Conversation deleted"}


# ==================== HEALTH CHECK ====================

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "mongodb": "connected" if client else "disconnected",
            "whisper": "configured" if os.environ.get('EMERGENT_LLM_KEY') else "not configured",
            "gemini": "configured" if os.environ.get('EMERGENT_LLM_KEY') else "not configured",
            "tts": "configured" if os.environ.get('GOOGLE_TTS_API_KEY') else "not configured"
        }
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
