# Counselling_Services - ChatBot

# IITGN Counseling Chatbot


## Prerequisites
- Node.js 18+
- (Optional) [Ollama](https://ollama.ai) running locally if you want LLM-enhanced answers (default URL `http://localhost:11434`, model `mistral`).

## Install
```bash
npm install
```

## Prepare FAQs
1) Edit `faqs_raw.json` as needed.  
2) Precompute embeddings (downloads the model on first run):
```bash
node precompute_embeddings.js
```
This writes `faqs_with_embeddings.json` used by the server.

## Run the server
```bash
node server.js
```
Default port: `3000`. The server will load embeddings and the local embedding model (`Xenova/bge-small-en-v1.5`).

### Env vars (optional)
Set in a `.env` file or your shell:
- `PORT` – server port (default `3000`)
- `OLLAMA_BASE_URL` – e.g., `http://localhost:11434`
- `OLLAMA_MODEL` – e.g., `mistral`

## Try the demo UI
Open `chat.html` in your browser (served statically by the server).  
Type a question; you may see:
- `answer` mode: direct answer (optionally enhanced by LLM)
- `suggest` mode: buttons with likely FAQ matches
- `crisis` mode: crisis safety response
- `fallback` mode: low-confidence reply



## Notes
- Embedding model runs locally via `@xenova/transformers`; first run downloads weights.
- LLM call is optional; if Ollama is unreachable, it falls back to the top FAQ answer.***

