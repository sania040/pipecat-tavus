# Pipecat Tavus AI Avatar Agent

This is an example of a Tavus avatar that uses the Pipecat SDK to create an AI agent that can be used in a Daily room.

## Installation

```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup

In this project's directory, run the following command to copy the `.env.example` file to `.env`:

```console
cp .env.example .env
```

Edit the `.env` file with your own values.

### Tavus

Visit https://tavus.io to get your `TAVUS_API_KEY` and `TAVUS_REPLICA_ID`.

### OpenAI

Visit https://platform.openai.com to get your `OPENAI_API_KEY`.

### Cartesia

Visit https://cartesia.ai to get your `CARTESIA_API_KEY`.

### Deepgram

Visit https://deepgram.com to get your `DEEPGRAM_API_KEY`.

## Usage

Run the following command to start the agent:
```console
python tavus_agent.py
```

Then, from the logs in above, find and copy the `URL` of the room and paste it into your browser. It should look something like: `Joining https://tavus.daily.co/<room-id>`. Follow that link in the browser to join the room and begin speaking with your avatar.


## References

- [Pipecat AI Documentation](https://docs.pipecat.ai/server/services/video/tavus)
- [Pipecat AI GitHub Example](https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/21-tavus-layer.py)