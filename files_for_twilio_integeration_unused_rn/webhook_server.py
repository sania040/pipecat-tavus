import asyncio
import os
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
import socketio
import uvicorn
from dotenv import load_dotenv
from loguru import logger
import sys

# Import your transport implementation
from twilio_transport import TwilioTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Setup logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Load environment variables
load_dotenv(override=True)

# Create FastAPI app
app = FastAPI(title="Pipecat Video Agent")

# Create Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app)

# Initialize transport
transport = TwilioTransport(
    bot_name="Pipecat Video Agent",
    vad_enabled=True,
    vad_analyzer=SileroVADAnalyzer(),
    account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
    auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
    webhook_url=os.getenv("TWILIO_WEBHOOK_URL")
)

# Share the socketio instance with transport
transport.sio = sio

# Store active pipeline runners
active_sessions = {}

@app.get("/")
async def root():
    return {"status": "online", "service": "Pipecat Video Agent"}

@app.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """Handle incoming Twilio voice calls"""
    form_data = await request.form()
    data = dict(form_data)
    
    logger.info(f"Incoming call: {data}")
    
    # Handle the call in the transport
    twiml_response = await transport.handle_incoming_call(data)
    
    # Start or join a pipeline for this call
    call_sid = data.get('CallSid')
    if call_sid and call_sid not in active_sessions:
        # Launch a new pipeline in the background
        asyncio.create_task(start_pipeline_for_call(call_sid))
    
    return Response(content=str(twiml_response), media_type="application/xml")

@app.post("/twilio/status")
async def twilio_status_webhook(request: Request):
    """Handle Twilio status callbacks"""
    form_data = await request.form()
    data = dict(form_data)
    
    logger.info(f"Call status update: {data}")
    
    # Process status update in the transport
    await transport.handle_status_update(data)
    
    # Clean up resources if call has ended
    call_sid = data.get('CallSid')
    call_status = data.get('CallStatus')
    
    if call_sid and call_status in ['completed', 'failed', 'busy', 'no-answer', 'canceled']:
        await cleanup_session(call_sid)
    
    return Response(content="", media_type="text/plain")

@app.post("/twilio/audio")
async def twilio_audio_webhook(request: Request):
    """Handle Twilio audio stream events"""
    form_data = await request.form()
    data = dict(form_data)
    
    # Process audio stream event in the transport
    await transport.handle_audio_event(data)
    
    return Response(content="", media_type="text/plain")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication with the frontend"""
    await websocket.accept()
    
    # Register this connection with the session
    if session_id in active_sessions:
        active_sessions[session_id]['websockets'] = active_sessions[session_id].get('websockets', []) + [websocket]
    
    try:
        while True:
            data = await websocket.receive_json()
            # Process messages from the frontend
            if session_id in active_sessions and 'pipeline' in active_sessions[session_id]:
                pipeline = active_sessions[session_id]['pipeline']
                await pipeline.process_frontend_message(data)
    except WebSocketDisconnect:
        # Remove this websocket from the session
        if session_id in active_sessions and 'websockets' in active_sessions[session_id]:
            if websocket in active_sessions[session_id]['websockets']:
                active_sessions[session_id]['websockets'].remove(websocket)

@sio.event
async def connect(sid, environ):
    """Handle Socket.IO connection"""
    logger.info(f"Socket.IO client connected: {sid}")

@sio.event
async def disconnect(sid):
    """Handle Socket.IO disconnection"""
    logger.info(f"Socket.IO client disconnected: {sid}")

async def start_pipeline_for_call(call_sid):
    """Initialize and start a pipeline for a new call"""
    from pipecat.pipeline import Pipeline
    from pipecat.audio.transcription import WhisperTranscriber
    from pipecat.llm.agent import VideoAgentLLM
    
    logger.info(f"Starting pipeline for call: {call_sid}")
    
    
    # Create pipeline components
    transcriber = WhisperTranscriber(
        model=os.getenv("WHISPER_MODEL", "base"),
        buffer_size=os.getenv("WHISPER_BUFFER_SIZE", 5)
    )
    
    # Initialize the LLM-based agent
    agent = VideoAgentLLM(
        model=os.getenv("LLM_MODEL", "gpt-4-0613"),
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful video call assistant.")
    )
    
    # Create the pipeline
    pipeline = Pipeline(
        transcriber=transcriber,
        agent=agent,
        transport=transport,
        session_id=call_sid
    )
    
    # Store the pipeline in active sessions
    active_sessions[call_sid] = {
        'pipeline': pipeline,
        'start_time': asyncio.get_event_loop().time(),
        'websockets': []
    }
    
    # Start the pipeline
    try:
        await pipeline.start()
        
        # Run the pipeline until completion
        await pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        # Cleanup after pipeline ends
        await cleanup_session(call_sid)

async def cleanup_session(call_sid):
    """Clean up resources for a finished call"""
    if call_sid in active_sessions:
        logger.info(f"Cleaning up session for call: {call_sid}")
        
        # Close any active websockets
        if 'websockets' in active_sessions[call_sid]:
            for ws in active_sessions[call_sid]['websockets']:
                try:
                    await ws.close()
                except Exception as e:
                    logger.error(f"Error closing websocket: {e}")
        
        # Stop the pipeline if it's still running
        if 'pipeline' in active_sessions[call_sid]:
            try:
                await active_sessions[call_sid]['pipeline'].stop()
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")
        
        # Remove the session
        del active_sessions[call_sid]

# Broadcast updates to all connected websockets for a session
async def broadcast_to_session(session_id, event, data):
    """Send updates to all connected frontend clients"""
    if session_id in active_sessions and 'websockets' in active_sessions[session_id]:
        for ws in active_sessions[session_id]['websockets']:
            try:
                await ws.send_json({"event": event, "data": data})
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")

# Register the broadcast function with the transport
transport.broadcast_func = broadcast_to_session

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Pipecat Video Agent on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )