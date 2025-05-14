import asyncio
import os
import sys
from typing import Any, Mapping

import aiohttp
from fastapi import FastAPI, Request, Response
import socketio
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tavus.video import TavusVideoService

from twilio_transport import TwilioTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def create_pipeline(transport=None):
    """Create and start a Pipecat pipeline with the provided transport"""
    async with aiohttp.ClientSession() as session:
        # Initialize Tavus for video generation
        tavus = TavusVideoService(
            api_key=os.getenv("TAVUS_API_KEY"),
            replica_id=os.getenv("TAVUS_REPLICA_ID"),
            session=session,
        )
        
        # Get persona name and initialize room
        persona_name = await tavus.get_persona_name()
        room_url = await tavus.initialize()
        
        logger.info(f"Initialized Tavus with persona: {persona_name}, room URL: {room_url}")

        # Create transport if not provided
        if transport is None:
            transport = TwilioTransport(
                bot_name="Pipecat Video Agent",
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            )

        # Initialize speech-to-text service
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Initialize text-to-speech service
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="156fb8d2-335b-4950-9cb3-a2d33befec77",  # Default voice ID
        )

        # Initialize the LLM service
        llm = OpenAILLMService(model="gpt-4o-mini")

        # Create the conversation context
        messages = [
            {
                "role": "system",
                "content": """You are a helpful and friendly video agent named Luna. 
                Your responses will be converted to video and audio, so keep your answers concise and engaging.
                You're talking to someone via a video call. Be personable and conversational.
                Avoid using technical jargon unless the user brings it up first.
                If the user asks about the technology powering this call, you can explain you're a Pipecat-powered
                video agent using Tavus for video generation, Deepgram for speech recognition, 
                Cartesia for voice synthesis, and GPT for conversation.""",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Create the pipeline
        pipeline = Pipeline(
            [
                transport.input(),                # Transport user input
                stt,                              # Speech-to-Text
                context_aggregator.user(),        # Add user responses to context
                llm,                              # Language model processing
                tts,                              # Text-to-Speech
                tavus,                            # Video generation
                transport.output(),               # Transport output to user
                context_aggregator.assistant(),   # Add assistant responses to context
            ]
        )

        # Create the pipeline task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(
            transport: TwilioTransport, participant: Mapping[str, Any]
        ) -> None:
            # Ignore the Tavus replica's microphone if present
            if participant.get("info", {}).get("userName", "") == persona_name:
                logger.debug(f"Ignoring {participant['id']}'s microphone")
                await transport.update_subscriptions(
                    participant_settings={
                        participant["id"]: {
                            "media": {"microphone": "unsubscribed"},
                        }
                    }
                )

            if participant.get("info", {}).get("userName", "") != persona_name:
                # Kick off the conversation
                logger.info(f"New participant joined: {participant.get('info', {}).get('userName', 'Unknown')}")
                messages.append(
                    {"role": "system", "content": "Introduce yourself warmly and ask how you can help them today."}
                )
                await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant.get('info', {}).get('userName', 'Unknown')}, reason: {reason}")
            await task.cancel()

        # Run the pipeline
        runner = PipelineRunner()
        return task

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

    return Response(content=twiml_response, media_type="application/xml")

@app.post("/twilio/status")
async def twilio_status_webhook(request: Request):
    """Handle Twilio call status callbacks"""
    form_data = await request.form()
    data = dict(form_data)

    call_sid = data.get('CallSid')
    status = data.get('CallStatus')

    logger.info(f"Call status update: {call_sid} -> {status}")

    if status in ['completed', 'failed', 'busy', 'no-answer', 'canceled']:
        # Call ended, cleanup
        if call_sid in transport.calls:
            participant_id = transport.calls[call_sid].get('participant_id')
            if participant_id:
                await transport._trigger_event(
                    "on_participant_left", 
                    transport, 
                    transport.participants.get(participant_id, {}), 
                    f"call_{status}"
                )
                
        # Clean up pipeline if it exists
        if call_sid in active_sessions:
            # Cancel the pipeline task
            if 'task' in active_sessions[call_sid]:
                try:
                    await active_sessions[call_sid]['task'].cancel()
                except:
                    pass
            del active_sessions[call_sid]

    return Response(content="OK")

@app.post("/call")
async def make_call(request: Request):
    """API endpoint to initiate a call"""
    data = await request.json()
    to_number = data.get('to')
    from_number = data.get('from', None)

    if not to_number:
        return {"status": "error", "message": "Missing 'to' phone number"}

    call_sid = await transport.make_call(to_number, from_number)

    if call_sid:
        return {"status": "success", "call_sid": call_sid}
    else:
        return {"status": "error", "message": "Failed to initiate call"}

async def start_pipeline_for_call(call_sid):
    """Start a Pipecat pipeline for a specific call"""
    logger.info(f"Starting pipeline for call {call_sid}")

    try:
        # Create and start the pipeline
        pipeline_task = await create_pipeline(transport)
        
        # Store reference to the running pipeline
        active_sessions[call_sid] = {
            'task': pipeline_task,
            'start_time': asyncio.get_event_loop().time()
        }
        
        # Start the pipeline runner
        runner = PipelineRunner()
        await runner.run(pipeline_task)
        
    except Exception as e:
        logger.error(f"Error in pipeline for call {call_sid}: {e}")
    finally:
        # Cleanup when done
        if call_sid in active_sessions:
            del active_sessions[call_sid]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)