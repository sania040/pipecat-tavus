import asyncio
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from loguru import logger

from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StopTaskFrame,
    ErrorFrame,
)
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import Frame, FrameSource, FrameSink
from pipecat.pipeline.task import FrameInfo
from pipecat.transports.base_transport import BaseTransport

# Import Twilio Client for voice calls
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.twiml.messaging_response import MessagingResponse
import socketio
import uuid


class TwilioAudioSource(FrameSource):
    """Audio source that receives audio from a Twilio call"""

    def __init__(
        self,
        transport: "TwilioTransport",
        vad_enabled: bool = False,
        vad_analyzer: Optional[VADAnalyzer] = None,
    ):
        super().__init__()
        self.transport = transport
        self.vad_enabled = vad_enabled
        self.vad_analyzer = vad_analyzer
        self.participants = {}
        self.buffer = asyncio.Queue()

    async def next_frame(self) -> Optional[InputAudioRawFrame]:
        """Get the next audio frame from the buffer"""
        try:
            frame = await self.buffer.get()
            return frame
        except asyncio.CancelledError:
            logger.debug("TwilioAudioSource.next_frame cancelled")
            return None
        except Exception as e:
            logger.error(f"Error in TwilioAudioSource.next_frame: {e}")
            return None

    async def add_audio_data(
        self,
        participant_id: str,
        audio_data: bytes,
        sample_rate: int = 16000,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Add audio data to the buffer for processing"""
        # Skip if the participant is muted
        if participant_id in self.participants and self.participants[participant_id].get("muted", False):
            return

        # Create audio frame
        frame = InputAudioRawFrame(
            audio=audio_data,
            sample_rate=sample_rate,
            num_channels=1,  # Assuming mono audio
        )
        await self.buffer.put(frame)
        logger.debug(f"Added InputAudioRawFrame for participant {participant_id}")


class TwilioAudioSink(FrameSink):
    """Audio sink that sends audio to a Twilio call"""

    def __init__(self, transport: "TwilioTransport"):
        super().__init__()
        self.transport = transport

    async def process_frame(self, frame: OutputAudioRawFrame) -> None:
        """Process an audio frame by sending it to all active participants"""
        if not isinstance(frame, OutputAudioRawFrame):
            logger.warning(f"TwilioAudioSink received non-OutputAudioRawFrame: {type(frame)}")
            return

        await self.transport.send_audio_to_participants(frame)


class TwilioTransport(BaseTransport):
    """Transport for Twilio voice calls"""

    def __init__(
        self,
        bot_name: str,
        vad_enabled: bool = False,
        vad_analyzer: Optional[VADAnalyzer] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        super().__init__()
        self.bot_name = bot_name
        self.vad_enabled = vad_enabled
        self.vad_analyzer = vad_analyzer
        self.participants = {}
        self.calls = {}

        # Twilio client setup
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.webhook_url = webhook_url or os.getenv("TWILIO_WEBHOOK_URL")

        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            logger.warning("Twilio credentials not provided. Some features will be unavailable.")
            self.client = None

        # Setup socket.io for real-time audio streaming
        self.sio = socketio.AsyncServer(async_mode="asyncio")

        # Event handlers
        self._event_handlers = {}

        # Create audio source and sink
        self.audio_input = TwilioAudioSource(self, vad_enabled, vad_analyzer)
        self.audio_output = TwilioAudioSink(self)

        # Setup socket.io events
        self._setup_socketio_events()

    def _setup_socketio_events(self):
        """Setup Socket.IO event handlers"""

        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")

        @self.sio.event
        async def disconnect(sid):
            # Find participant by sid and trigger left event
            participant_id = None
            for pid, data in self.participants.items():
                if data.get("sid") == sid:
                    participant_id = pid
                    break

            if participant_id:
                await self._trigger_event(
                    "on_participant_left", self, self.participants[participant_id], "disconnected"
                )
                del self.participants[participant_id]

            logger.info(f"Client disconnected: {sid}")

        @self.sio.event
        async def audio_data(sid, data):
            # Find participant by sid
            participant_id = None
            for pid, pdata in self.participants.items():
                if pdata.get("sid") == sid:
                    participant_id = pid
                    break

            if participant_id:
                await self.audio_input.add_audio_data(
                    participant_id,
                    data["audio"],
                    data.get("sample_rate", 16000),
                )

    def input(self) -> FrameSource:
        """Get the audio input source"""
        return self.audio_input

    def output(self) -> FrameSink:
        """Get the audio output sink"""
        return self.audio_output

    async def send_audio_to_participants(self, frame: OutputAudioRawFrame):
        """Send audio data to all connected participants"""
        if not self.participants:
            return

        audio_data = frame.audio

        # Broadcast to all connected participants via Socket.IO
        for participant_id, data in self.participants.items():
            if not data.get("muted", False) and "sid" in data:
                try:
                    await self.sio.emit(
                        "audio",
                        {
                            "audio": audio_data,
                            "sample_rate": frame.sample_rate,
                            "source": self.bot_name,
                        },
                        room=data["sid"],
                    )
                except Exception as e:
                    logger.error(f"Error sending audio to participant {participant_id}: {e}")

    async def update_subscriptions(self, participant_settings):
        """Update participant subscription settings"""
        # This method would update media subscriptions for participants
        # In a real implementation, this might involve Twilio API calls
        for participant_id, settings in participant_settings.items():
            if participant_id in self.participants:
                # Apply the settings to our local participant data
                media_settings = settings.get("media", {})
                if "microphone" in media_settings:
                    self.participants[participant_id]["muted"] = (
                        media_settings["microphone"] == "unsubscribed"
                    )
                logger.debug(f"Updated subscription for {participant_id}: {settings}")
        return True

    async def handle_incoming_call(self, data):
        """Handle incoming Twilio call webhook data"""
        call_sid = data.get("CallSid")
        from_number = data.get("From")
        
        if not call_sid:
            logger.error("No CallSid provided in incoming call data")
            return "<Response><Say>Error processing call</Say></Response>"
        
        # Generate a unique participant ID for this call
        participant_id = f"call_{call_sid}"
        
        # Store call information
        self.calls[call_sid] = {
            "participant_id": participant_id,
            "from": from_number,
            "status": "ringing",
            "created_at": asyncio.get_event_loop().time(),
        }
        
        # Add the participant
        self.participants[participant_id] = {
            "id": participant_id,
            "info": {
                "userName": f"Caller {from_number}",
                "callSid": call_sid,
            },
            "muted": False,
            "sid": None,  # Will be set when they connect via socket.io
        }
        
        # Trigger participant joined event
        await self._trigger_event("on_participant_joined", self, self.participants[participant_id])
        
        # Generate TwiML response
        response = VoiceResponse()
        response.say("Welcome to the Pipecat Video Agent. Please wait while we connect you.")
        
        # If we have a webhook URL, add a <Connect> to our WebSocket
        if self.webhook_url:
            start = Start()
            start.stream(url=f"{self.webhook_url}/socket.io/")
            response.append(start)
        
        return str(response)

    async def make_call(self, to_number, from_number=None):
        """Make an outbound call using Twilio"""
        if not self.client:
            logger.error("Cannot make outbound call: Twilio client not initialized")
            return None
            
        if not from_number:
            # Use the default Twilio number if one is not provided
            from_number = os.getenv("TWILIO_PHONE_NUMBER")
            
        if not from_number:
            logger.error("No 'from' number provided for outbound call")
            return None
            
        try:
            # Make the call via Twilio API
            call = self.client.calls.create(
                to=to_number,
                from_=from_number,
                url=f"{self.webhook_url}/twilio/voice",
                status_callback=f"{self.webhook_url}/twilio/status",
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            )
            
            logger.info(f"Initiated outbound call to {to_number}: {call.sid}")
            return call.sid
            
        except Exception as e:
            logger.error(f"Failed to make outbound call: {e}")
            return None

    def event_handler(self, event_name):
        """Decorator for registering event handlers"""
        def decorator(func):
            if event_name not in self._event_handlers:
                self._event_handlers[event_name] = []
            self._event_handlers[event_name].append(func)
            return func
        return decorator

    async def _trigger_event(self, event_name, *args, **kwargs):
        """Trigger registered event handlers"""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    await handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler {event_name}: {e}")