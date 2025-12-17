# IITGN Counselling Services Chatbot

A lightweight Node.js server that answers IIT Gandhinagar Counselling Services FAQs using hybrid retrieval (semantic + keyword) and a local LLM (Ollama). Embeddings are computed via `@xenova/transformers` in a worker pool using Piscina.

## Prerequisites
- Node.js 18+ (recommended 20+)
- Internet access (first run downloads the embedding model)
- Optional: [Ollama](https://ollama.com) installed and running if you want LLM answers locally
- A `faqs_with_embeddings.json` file in this folder (already included)

## Quick Start

### 1) Install dependencies
```bash
npm install

```

### 2) (Optional) Install Ollama and pull a model
If you plan to use a local LLM via Ollama (default model: `mistral`):
```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download and install from https://ollama.com/download/windows

# After install, pull the model
ollama pull mistral:7b-instruct

```

You can also use the provided scripts (if Ollama is installed):
```bash
npm run pull-model     # pulls mistral
npm run check-ollama   # lists installed models
```

### 3) Configure environment (optional)
Create a `.env` file in this folder if you want to customize defaults.
```env
# Server
PORT=3000

# Ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=mistral
```
If you deploy Ollama on a different machine, set `OLLAMA_URL` to that host. Example: `http://<server-ip>:11434/api/generate`.

### 4) Start the server
```bash
npm start
```
The server starts on `http://localhost:3000` by default.



Notes:
- Requests are rate-limited to 30/minute per IP
- Messages longer than 500 characters are rejected politely
- Crisis-related keywords trigger a supportive safety response

## Project Structure (this folder)
- `server.js` — Express server, retrieval logic, Ollama call
- `embeddingWorker.js` — Piscina-compatible worker that computes query embeddings
- `faqs_with_embeddings.json` — FAQ corpus with precomputed embeddings
- `package.json` — Scripts and dependencies

## How It Works
- The server embeds the user query in a background worker (Piscina) using `Xenova/bge-small-en-v1.5`.
- It performs a hybrid search over the FAQ corpus: semantic (cosine similarity) + keyword (Fuse.js).
- Top results construct a prompt for the LLM (Ollama). If the LLM can’t answer from the provided context, a controlled fallback reply is returned.



## Production Tips
- Pin your Node version (e.g., via `.nvmrc` or runtime image)
- Restrict CORS to trusted origins
- Put the app behind HTTPS (reverse proxy like Nginx/Caddy)
- Add structured logging (e.g., pino/winston) and health checks
- Tune Piscina thread counts for your CPU/traffic profile

## Useful Scripts
```bash
npm start          # Start the server
npm run dev        # Alias for start
npm run pull-model # Pull the default Ollama model (mistral)
npm run check-ollama # List installed Ollama models
```

## Test With curl
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What services are offered by counselling?"}'
```

