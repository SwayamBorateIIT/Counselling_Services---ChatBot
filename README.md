# IITGN Counselling Services Chatbot

A lightweight Node.js server that answers IIT Gandhinagar Counselling Services FAQs using hybrid retrieval (semantic + keyword) and Groq API. Embeddings are computed via `@xenova/transformers` in a worker pool using Piscina.

## Prerequisites
- Node.js 18+ (recommended 20+)
- Internet access
- A Groq API key (get it from [https://console.groq.com](https://console.groq.com))
- A `faqs_with_embeddings.json` file in this folder (already included)

## Quick Start

### 1) Install dependencies
```bash
npm install
```

### 2) Get a Groq API Key
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Create an API key
4. Copy your API key

### 3) Configure environment
Create a `.env` file in this folder with your Groq API key:
```env
PORT=3000
LLM_PROVIDER=groq
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768
EMERGENCY_NUMBER=+91 7888317528
```

**Available Groq Models:**
- `mixtral-8x7b-32768` (recommended, fastest)
- `llama2-70b-4096`
- `gemma-7b-it`
- `llama2-70b-chat`

### 4) Start the server
```bash
npm start
```
The server starts on `http://localhost:3000` by default.

## Notes
- Requests are rate-limited to 30/minute per IP
- Messages longer than 500 characters are rejected politely
- Crisis-related keywords trigger a supportive safety response
- Streaming responses for real-time feedback

## Project Structure (this folder)
- `server.js` — Express server, retrieval logic, Groq API integration
- `embeddingWorker.js` — Piscina-compatible worker that computes query embeddings
- `faqs_with_embeddings.json` — FAQ corpus with precomputed embeddings
- `package.json` — Scripts and dependencies
- `.env` — Configuration (API key, model, port)

## How It Works
- The server embeds the user query in a background worker (Piscina) using `Xenova/bge-small-en-v1.5`.
- It performs a hybrid search over the FAQ corpus: semantic (cosine similarity) + keyword (Fuse.js).
- Top results construct a prompt for the Groq API. If the LLM can't answer from the provided context, a controlled fallback reply is returned.

## Production Tips
- Pin your Node version (e.g., via `.nvmrc` or runtime image)
- Use environment variables for sensitive data (never commit `.env`)
- Restrict CORS to trusted origins
- Put the app behind HTTPS (reverse proxy like Nginx/Caddy)
- Add structured logging (e.g., pino/winston) and health checks
- Monitor API quota usage on Groq console

## Useful Scripts
```bash
npm start          # Start the server
npm run dev        # Alias for start
```

## Test With curl
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What services are offered by counselling?"}'
```
