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
import io

# OpenAI for Whisper
from openai import OpenAI

# TWB Voice Hausa TTS (Coqui TTS)
from TTS.api import TTS
from huggingface_hub import hf_hub_download
import json
import scipy.io.wavfile as wavfile
import numpy as np

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

# TWB Voice Hausa TTS model (lazy loading)
twb_tts = None
twb_temp_config = None

def get_twb_tts():
    """Load TWB Voice Hausa TTS model (CLEAR-Global/TWB-Voice-Hausa-TTS-1.0)"""
    global twb_tts, twb_temp_config
    if twb_tts is None:
        logger.info("Loading TWB Voice Hausa TTS model (CLEAR-Global/TWB-Voice-Hausa-TTS-1.0)...")
        
        # Download model files from Hugging Face
        model_name = "CLEAR-Global/TWB-Voice-Hausa-TTS-1.0"
        
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Download required files
        model_path = hf_hub_download(model_name, "best_model_498283.pth")
        speakers_file = hf_hub_download(model_name, "speakers.pth")
        language_ids_file = hf_hub_download(model_name, "language_ids.json")
        d_vector_file = hf_hub_download(model_name, "d_vector.pth")
        config_se_file = hf_hub_download(model_name, "config_se.json")
        model_se_file = hf_hub_download(model_name, "model_se.pth")
        
        # Update config paths
        config["speakers_file"] = speakers_file
        config["language_ids_file"] = language_ids_file
        config["d_vector_file"] = [d_vector_file]
        config["model_args"]["speakers_file"] = speakers_file
        config["model_args"]["language_ids_file"] = language_ids_file
        config["model_args"]["d_vector_file"] = [d_vector_file]
        config["model_args"]["speaker_encoder_config_path"] = config_se_file
        config["model_args"]["speaker_encoder_model_path"] = model_se_file
        
        # Save updated config to temp file
        twb_temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, twb_temp_config, indent=2)
        twb_temp_config.close()
        
        # Load TTS model
        twb_tts = TTS(model_path=model_path, config_path=twb_temp_config.name)
        logger.info("TWB Voice Hausa TTS model loaded successfully!")
    
    return twb_tts

# Create the main app without a prefix
app = FastAPI(title="Menene - Hausa Conversational AI (TWB Voice TTS)")

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
    language: str = "ha"  # Hausa language code
    speaker: str = "spk_f_1"  # Options: spk_f_1 (female), spk_m_1 (male), spk_m_2 (male)


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


# ==================== TEXT TO SPEECH ENDPOINT (TWB Voice Hausa TTS) ====================

@api_router.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using TWB Voice Hausa TTS (CLEAR-Global/TWB-Voice-Hausa-TTS-1.0)
    Multi-speaker model with 3 speakers: spk_f_1 (female), spk_m_1 (male), spk_m_2 (male)
    """
    try:
        logger.info(f"TTS request for text: {request.text[:50]}...")
        
        # Validate speaker
        valid_speakers = ["spk_f_1", "spk_m_1", "spk_m_2"]
        speaker = request.speaker if request.speaker in valid_speakers else "spk_f_1"
        
        # Check cache first
        cached_audio = await db.audio_cache.find_one({
            "text": request.text.lower(),
            "language": "ha",
            "voice": f"twb-voice-{speaker}"
        })
        
        if cached_audio:
            logger.info("Returning cached audio")
            return {
                "success": True,
                "audio_content": cached_audio["audio_content"],
                "cached": True,
                "tts_engine": "twb-voice-hausa-tts",
                "speaker": speaker
            }
        
        # Load TWB Voice TTS model
        tts = get_twb_tts()
        
        # Generate speech (TWB Voice requires lowercase input)
        text_lower = request.text.lower()
        wav = tts.synthesizer.tts(text=text_lower, speaker_name=speaker)
        
        # Convert to numpy array
        wav_array = np.array(wav, dtype=np.float32)
        
        # Get sample rate (24 kHz for TWB Voice)
        sample_rate = tts.synthesizer.output_sample_rate
        
        # Save to WAV bytes
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sample_rate, wav_array)
        wav_bytes = wav_buffer.getvalue()
        
        # Convert to base64
        audio_content = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Cache the audio
        await db.audio_cache.insert_one({
            "text": text_lower,
            "language": "ha",
            "voice": f"twb-voice-{speaker}",
            "audio_content": audio_content,
            "created_at": datetime.utcnow()
        })
        
        logger.info(f"TTS audio generated and cached using TWB Voice Hausa TTS (speaker: {speaker})")
        
        return {
            "success": True,
            "audio_content": audio_content,
            "cached": False,
            "tts_engine": "twb-voice-hausa-tts",
            "speaker": speaker,
            "sample_rate": sample_rate
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
            "tts": "twb-voice-hausa-tts (CLEAR-Global/TWB-Voice-Hausa-TTS-1.0)"
        },
        "tts_engine": "TWB Voice Hausa TTS",
        "tts_speakers": ["spk_f_1 (female)", "spk_m_1 (male)", "spk_m_2 (male)"]
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
