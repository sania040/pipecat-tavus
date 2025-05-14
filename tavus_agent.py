import sys
sys.path.insert(0, r"D:/projects/pipecat-tavus/pipecat/src")


import asyncio
import os
import sys
import aiohttp
from dotenv import load_dotenv
from loguru import logger
import sounddevice as sd
import numpy as np

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService, OpenAIUserContextAggregator, OpenAIAssistantContextAggregator
from pipecat.services.tavus.video import TavusVideoService
from pipecat.processors.async_generator import AsyncGeneratorProcessor
from pipecat.processors.frame_processor import FrameProcessor

class SimpleSerializer:
    async def serialize(self, frame):
        # Just pass through the frame without any changes
        return frame
        
    async def deserialize(self, data):
        # Just pass through the data without any changes
        return data


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class MicInput(AsyncGeneratorProcessor):
    def __init__(self):
        print("📣 MicInput initialized")
        super().__init__(serializer=SimpleSerializer())
        self._generator_started = False

    async def start(self):
        print("📣 MicInput.start() called")
        await super().start()
    
    async def process_frame(self, frame, direction):
        print(f"📣 MicInput.process_frame called with frame: {type(frame).__name__}")
        
        if not self._generator_started and hasattr(frame, 'name') and 'StartFrame' in frame.name:
            self._generator_started = True
            # Run the generator as a background task
            asyncio.create_task(self._run_generator())
            
        return await super().process_frame(frame, direction)
    
    async def _run_generator(self):
        print("🔄 Starting generator task...")
        async for frame in self.generator():
            print(f"⏩ Forwarding frame from generator: {frame.keys() if isinstance(frame, dict) else type(frame).__name__}")
            await self.push_frame(frame)

    async def generator(self):
        print("📣 MicInput.generator() called - starting audio capture")
        fs = 16000
        duration = 5
        
        # List available audio devices
        print("🎤 Available audio devices:")
        for i, device in enumerate(sd.query_devices()):
            print(f"  {i}: {device['name']} (in: {device['max_input_channels']}, out: {device['max_output_channels']})")
        
        print("🎙 Speak now — recording 5 seconds...")
        # Use float32 for better amplitude control
        audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        print("✅ Audio captured.")
        
        # Apply gain to increase volume (adjust as needed)
        gain = 15.0  # Increase volume by 15x
        audio = audio * gain
        
        # Clip to prevent distortion and convert to int16
        audio_array = np.int16(np.clip(audio * 32767, -32768, 32767)).flatten()
        
        print(f"🔍 Audio stats: length={len(audio_array)}, max={np.max(audio_array)}, min={np.min(audio_array)}")
        
        # Check if audio is still silent after gain
        if np.max(np.abs(audio_array)) < 1000:
            print("⚠ Warning: Audio volume is still very low even after applying gain.")
            print("💡 Try checking your system's microphone settings or connecting an external microphone.")
        
        yield {
            "audio": audio_array.tobytes(),
            "sample_rate": fs
        }
        
        print("✅ Audio frame yielded to pipeline.")


# Use FrameProcessor instead of Processor
class PrintOutput(FrameProcessor):
    async def process_frame(self, frame, *args, **kwargs):
        print(f"🔊 Output frame type: {type(frame).__name__}")
        print(f"🔊 Output frame content: {frame}")
        return frame  

# Create a simple context aggregator that works around the compatibility issue
class SimpleContextAggregator(FrameProcessor):
    def __init__(self, messages, role="user"):
        super().__init__()
        self.messages = messages
        self.role = role

    async def process_frame(self, frame, direction):
        # Check if the frame contains text from transcription
        if isinstance(frame, dict) and "text" in frame:
            print(f"🗣 User said: {frame['text']}")
            
            # Add the user's message to the context
            if self.role == "user":
                # For user context, we're receiving text from STT
                self.messages.append({"role": "user", "content": frame["text"]})
            
        # For assistant context, we're receiving the response from LLM
        elif self.role == "assistant" and isinstance(frame, dict) and "message" in frame:
            content = frame["message"]["content"]
            print(f"🤖 Assistant response: {content}")
            self.messages.append({"role": "assistant", "content": content})
        
        return frame

# Change the context aggregator setup in the main function
async def main():
    print("🚀 Starting Tavus Agent pipeline...")
    async with aiohttp.ClientSession() as session:
        tavus = TavusVideoService(
            api_key=os.getenv("TAVUS_API_KEY"),
            replica_id=os.getenv("TAVUS_REPLICA_ID"),
            session=session,
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="156fb8d2-335b-4950-9cb3-a2d33befec77",
        )

        llm = OpenAILLMService(model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant named Luna. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        # Create the context without using OpenAILLMContext
        # Instead, create the aggregators directly and pass the messages
        user_context = SimpleContextAggregator(messages=messages, role="user")
        assistant_context = SimpleContextAggregator(messages=messages, role="assistant")

        pipeline = Pipeline([
            MicInput(),                      # 🎙 Mic Input
            stt,                             # 🧠 STT
            user_context,                    # 📥 LLM context (user)
            llm,                             # 🤖 LLM
            tts,                             # 🔊 TTS
            tavus,                           # 🎥 Tavus video
            PrintOutput(),                   # 🖨 Console print
            assistant_context,               # 📤 LLM context (bot)
        ])
        # pipeline.set_name("Tavus Agent")

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

        # Modify the PipelineRunner instantiation to handle Windows signal limitations
        if sys.platform == 'win32':
            # Windows-specific implementation
            # Monkey patch the _setup_sigint method to avoid the NotImplementedError
            original_setup_sigint = PipelineRunner._setup_sigint
            PipelineRunner._setup_sigint = lambda self: None
            
            runner = PipelineRunner()
            
            # Restore the original method after creation
            PipelineRunner._setup_sigint = original_setup_sigint
        else:
            # Standard implementation for Unix-like systems
            runner = PipelineRunner()
            
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
