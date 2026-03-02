# ADK Personal Assistant

An AI-powered personal assistant built with [Google Agent Development Kit (ADK)](https://github.com/google/adk-python).  
It integrates **Gmail**, **YouTube**, **Google Search**, and **token-cost tracking** into a single conversational agent you can run locally in minutes.

---

## Features

| Category | Tool | What it does |
|---|---|---|
| **Email** | `gmail_summary` | Summarised view of recent Primary-inbox messages with sender stats |
| | `get_email_content` | Full email body search by sender name / address / subject |
| **YouTube** | `search_youtube_videos` | Search & filter videos (date, duration, channel, sort order) |
| | `get_video_summary` | Transcript extraction → key points, books, quotes, references |
| | `check_video_transcripts` | List available transcript languages for any video |
| **Search** | Google Search | General web search powered by Google |
| **Cost** | `token_cost_calculator` | Per-interaction cost breakdown for Gemini models |
| | `batch_cost_estimator` | Estimate costs across N interactions |
| **Memory** | `update_memory` | Persist facts, preferences, and notes in a local markdown file |
| | `read_memory` | Recall stored memory (all sections or a specific one) |
| | `list_memory_sections` | List all top-level sections in memory |
| | `clear_memory_section` | Remove a section or sub-section from memory |
| **File Search** | `search_files` | Find files by partial name / glob pattern, bounded by time |
| | `search_file_content` | Full-text search inside files, optionally filtered by name and time |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | 3.13 recommended |
| **uv** (package manager) | Installed automatically in the dev container |
| **Gemini API access** | *Either* a free [Google AI Studio](https://aistudio.google.com/apikey) key **or** a GCP project with Vertex AI enabled **or** a local [Ollama](https://ollama.com) server |
| **YouTube Data API v3 key** | [Create one here](https://console.cloud.google.com/apis/credentials) |
| **Gmail OAuth credentials** | Only needed if you want the Gmail tools |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/adk_assistant.git
cd adk_assistant
```

### 2. Create a virtual environment & install dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate
uv pip install google-adk google-auth-oauthlib youtube-transcript-api
```

> **Dev Container users:** open this repo in VS Code with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension — dependencies install automatically.

### 3. Configure environment variables

```bash
cp personal_assistant/.env.example personal_assistant/.env
```

Open `personal_assistant/.env` and choose **one** of the two authentication options:

#### Option A — Free Gemini API Key (Google AI Studio)

```env
GOOGLE_GENAI_USE_VERTEXAI=false
GOOGLE_API_KEY=<your-api-key>
```

Get a free key at <https://aistudio.google.com/apikey>.

#### Option B — Vertex AI (GCP Project)

```env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_LOCATION=us-east4
```

Then authenticate:

```bash
gcloud auth application-default login
```

#### Option C — Local Ollama Model (no cloud required)

Run any model locally using [Ollama](https://ollama.com):

1. Install Ollama from <https://ollama.com/download>.
2. Pull a model (e.g. `llama3.2` — commercially free under the Meta Llama license):

```bash
ollama pull llama3.2
```

3. Set these variables in `.env` (and remove / comment out Options A and B):

```env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

> **Commercially-free models recommended for local/GCP use:**
> | Model | Pull command | Notes |
> |---|---|---|
> | `llama3.2` | `ollama pull llama3.2` | Meta Llama 3.2 — commercial use permitted |
> | `mistral` | `ollama pull mistral` | Mistral 7B — Apache 2.0 license |
> | `gemma2` | `ollama pull gemma2` | Google Gemma 2 — commercial use permitted |
> | `phi3` | `ollama pull phi3` | Microsoft Phi-3 Mini — MIT license |

#### YouTube & Gmail (both options)

```env
YOUTUBE_API_KEY=<your-youtube-data-api-key>
GMAIL_TOKEN_FILE=token.json
```


### 4. Set up Gmail OAuth (optional)

If you want the Gmail tools:

1. Create an **OAuth 2.0 Client ID** (Desktop app) in the [GCP Console → Credentials](https://console.cloud.google.com/apis/credentials).
2. Download the JSON and save it as `client_secret.json` in the project root.
3. Run the one-time auth flow:

```bash
python personal_assistant/onetime_auth_flow.py
```

This opens a browser window, asks you to log in, and saves `token.json` locally.

### 5. Run the assistant

```bash
adk web --host 0.0.0.0 --port 8001
```

Open <http://localhost:8001> in your browser.

---

## Project Structure

```
adk_assistant/
├── .devcontainer/              # VS Code dev container config
├── personal_assistant/
│   ├── __init__.py
│   ├── agent.py                # Agent definition & tool wiring
│   ├── onetime_auth_flow.py    # One-time Gmail OAuth helper
│   ├── .env.example            # Template — copy to .env
│   └── tools/
│       ├── gmail_summary.py        # Gmail read tools
│       ├── youtube_summary.py      # YouTube search & transcript tools
│       ├── token_cost_calculator.py# Cost estimation tools
│       ├── memory_manager.py       # Hierarchical local memory (markdown)
│       └── file_search.py          # Local file & content search
├── tests/
│   ├── test_gmail_tools.py
│   ├── test_youtube_tools.py
│   ├── test_token_cost_calculator.py
│   ├── test_retry_utils.py
│   ├── test_memory_manager.py      # Memory manager tests
│   └── test_file_search.py         # File search tests
├── .gitignore
└── README.md
```

> **Sensitive files you must provide yourself** (never committed):  
> `client_secret.json` · `token.json` · `personal_assistant/.env`

---

## Configuration Reference

### Changing the model (Google / Vertex AI)

Set the `AGENT_MODEL` env var in your `.env` for Gemini models:

```env
AGENT_MODEL=gemini-2.0-flash-001
```

Common Gemini choices:

| Model | Notes |
|---|---|
| `gemini-2.0-flash-001` | Default — fast & cheap |
| `gemini-2.0-flash-live-001` | Live/streaming voice mode (Vertex AI only, `us-east4`) |
| `gemini-1.5-flash-002` | Previous-gen flash |
| `gemini-1.5-pro` | Higher quality, higher cost |

### Recommended locally-hosted models (Ollama / 16 GB RAM)

These models run within a 16 GB RAM budget and provide solid chat +
summarisation quality. Licences vary (Apache 2.0, MIT, Meta Llama Community,
Gemma Terms), so please review the licence column and upstream terms to confirm
they fit your personal or commercial use case. They are also well-suited to a
16 GB GCP VM (e.g. `e2-standard-4` or `n1-standard-4`).

| Ollama tag | Licence | RAM needed | Strengths |
|---|---|---|---|
| `llama3.2` | Meta Llama 3.2 Community (free for most commercial use) | ~4 GB | Fast, general-purpose chat; excellent default choice |
| `llama3.1:8b` | Meta Llama 3.1 Community | ~6 GB | Stronger reasoning than 3.2, still fits 16 GB |
| `mistral` | Apache 2.0 | ~5 GB | Great for instruction following & summarisation |
| `gemma2:9b` | Gemma Terms (free for commercial use) | ~7 GB | Google model; strong code + reasoning |
| `qwen2.5:7b` | Apache 2.0 | ~5 GB | Multilingual; good at structured output |
| `phi3:mini` | MIT | ~2.5 GB | Very fast; good for low-latency voice |

> **Voice / streaming tip:** models with a smaller footprint (`phi3:mini`,
> `llama3.2`, `mistral`) respond faster and produce a more natural streaming
> voice experience because the first token arrives sooner.

To use one of these models set in your `.env`:

```env
USE_OLLAMA=true
OLLAMA_MODEL=mistral    # or any tag from the table above
```

### Region (Vertex AI only)

Ensure `GOOGLE_CLOUD_LOCATION` supports your chosen model.  
Gemini 2.0 Live requires `us-east4`.

### Memory file location

The agent stores memory in `~/.adk_assistant_memory.md` by default.
Override with the `MEMORY_FILE` env var:

```env
MEMORY_FILE=/path/to/my_memory.md
```

The file is plain markdown — you can read and edit it directly.

---

```
# Gmail
"Show me emails from the last 2 hours"
"Get the email from Jane about the quarterly report"

# YouTube
"Find latest AI podcasts from this week"
"Summarize this video: https://www.youtube.com/watch?v=VIDEO_ID"
"What books are mentioned in this podcast?"

# Cost
"How much would 1000 input tokens and 500 output tokens cost on gemini-2.0-flash-001?"
"Estimate batch cost for 100 interactions"

# Memory
"Remember that I prefer concise bullet-point answers"
"What do you know about me?"
"Forget the Notes section"

# File Search
"Find all PDF files in ~/Documents modified after 2024-01-01"
"Search for 'budget' inside text files under ~/projects"
"Find files named 'report' changed before 2024-06-01"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **Gmail auth fails** | Delete `token.json` and re-run `onetime_auth_flow.py` |
| **"YouTube API key not found"** | Set `YOUTUBE_API_KEY` in `.env` |
| **No transcript for a video** | Run `check_transcripts_tool` — some videos disable captions |
| **Model not found** | Confirm the model name and that your region supports it |
| **Ollama connection refused** | Make sure `ollama serve` is running; check `OLLAMA_BASE_URL` |
| **Ollama pull times out** | Large models can take minutes to download; re-run the agent or `ollama pull <model>` manually |

---

## Roadmap

- [x] Hierarchical local memory (markdown-backed)
- [x] Local file search by name, glob, and time bounds
- [x] Full-text search inside local files
- [x] Local Ollama model support (via LiteLLM)
- [ ] Chunked transcript processing for multi-hour podcasts
- [ ] Automatic per-query cost tracking
- [ ] Google Calendar integration
- [ ] GitHub-backed book/reading list from podcast mentions
- [ ] Multi-language transcript support
- [x] Streaming voice instructions (sentence-level TTS-optimised responses)
- [ ] Voice interface

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repo
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [Google ADK](https://github.com/google/adk-python) — Agent Development Kit
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — Transcript extraction
