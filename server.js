
import express from "express";
import fs from "fs";
import cors from "cors";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";
import fetch from "node-fetch";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const PORT = process.env.PORT || 3000;
const FAQ_FILE = "./faqs_with_embeddings.json";
const MODEL = "Xenova/bge-small-en-v1.5";

// LLM Provider config: Ollama (local)
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "mistral";

const AUTO_REPLY_THRESHOLD = 0.78;
const SUGGESTION_THRESHOLD = 0.60;
const TOP_K = 3;

// Short crisis keyword list — expand as needed
const CRISIS_KEYWORDS = [
  "suicide",
  "kill myself",
  "want to die",
  "hurt myself",
  "end my life",
  "can't go on",
  "i will kill myself"
].map((s) => s.toLowerCase());

function containsCrisis(text) {
  if (!text) return false;
  const t = text.toLowerCase();
  return CRISIS_KEYWORDS.some((k) => t.includes(k));
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  return Math.sqrt(dot(a, a));
}
function cosineSimilarity(a, b) {
  const denom = norm(a) * norm(b) || 1e-12;
  return dot(a, b) / denom;
}

// LLM function to enhance FAQ answers via selected provider
// Sends multiple FAQ candidates so the LLM can act as a lightweight re-ranker.
async function enhanceAnswerWithLLM(userMessage, faqCandidates) {
  const systemPrompt = `
    You are a counseling assistant for IIT Gandhinagar.
    Answer the user query using ONLY the provided FAQ context.
    Do NOT make up information or add external knowledge.
    If the FAQ context does not contain the answer, explicitly say you cannot answer.
    Keep the tone empathetic, professional, and brief (under 3 sentences).
    `
  const faqList = faqCandidates
    .map(
      (f, idx) =>
        `FAQ ${idx + 1} (score ${f.score?.toFixed(3) ?? "n/a"}):
Q: ${f.title}
A: ${f.a}`
    )
    .join("\n\n");

  const userPrompt = `A student asked: "${userMessage}"

Here are up to 3 potential FAQ matches. Use the one that best answers the student's question. If none apply, clearly say you don't know and suggest emailing cservices@iitgn.ac.in.

${faqList}

Provide a concise, empathetic response under 3 sentences, grounded strictly in the provided FAQs.`;

  try {
    const resp = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt }
        ],
        stream: false
      })
    });
    const data = await resp.json();
    if (!resp.ok) {
      console.warn("Ollama API error:", JSON.stringify(data, null, 2));
      return { enhanced: false, answer: faqAnswer };
    }
    const content = data?.message?.content || data?.choices?.[0]?.message?.content;
    if (content) return { enhanced: true, answer: content };
    console.warn("Ollama returned unexpected format:", JSON.stringify(data));
    // fall back to the top FAQ answer if no usable content
    return { enhanced: false, answer: faqCandidates[0]?.a };
  } catch (err) {
    console.error("Failed calling Ollama API:", err?.message || err);
    return { enhanced: false, answer: faqCandidates[0]?.a };
  }
}

const app = express();
app.use(express.json());
app.use(cors());
app.use(express.static(__dirname));

let faqs = [];
let extractor = null;

async function loadFaqs() {
  if (!fs.existsSync(FAQ_FILE)) {
    console.warn(`${FAQ_FILE} not found. Please run precompute_embeddings.js first.`);
    faqs = [];
    return;
  }
  faqs = JSON.parse(fs.readFileSync(FAQ_FILE, "utf8"));
  console.log(`Loaded ${faqs.length} FAQs.`);
}

// Initialize model once
async function init() {
  console.log("Initializing embedding extractor (model may download if first run)...");
  extractor = await pipeline("feature-extraction", MODEL);
  await loadFaqs();
  console.log("Ready.");
}

app.post("/api/faq-match", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message required" });
    }

    // 1) crisis check first
    if (containsCrisis(message)) {
      return res.json({
        mode: "crisis",
        reply:
          "If you are in immediate danger, please call emergency services or the emergency numbers listed on the counselling page. Would you like me to connect you to a counselor?"
      });
    }

    // 2) compute embedding for user message (local)
    if (!extractor) {
      extractor = await pipeline("feature-extraction", MODEL);
    }
    // BGE models expect an instruction prefix on queries for better retrieval quality
    const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
    const embOut = await extractor(`${QUERY_PREFIX}${message}`, {
      pooling: "mean",
      normalize: true
    });
    const qEmb = embOut.data ?? embOut;

    const scored = faqs
      .map((f) => {
        if (!f.embedding) return { id: f.id, title: f.title || f.q, a: f.a, score: -1 };
        const s = cosineSimilarity(qEmb, f.embedding);
        return { id: f.id, title: f.title || f.q, a: f.a, score: s };
      })
      .sort((x, y) => y.score - x.score);

    const best = scored[0];
    const topFaqs = scored.slice(0, TOP_K);

    const OUT_OF_SCOPE_THRESHOLD = 0.5; // Very low relevance means it's off-topic
    
    if (best && best.score >= AUTO_REPLY_THRESHOLD) {
      // Use LLM to enhance the answer for better relevance
      const { enhanced, answer } = await enhanceAnswerWithLLM(message, topFaqs);
      console.log(`✅ LLM Enhancement: ${enhanced ? "SUCCESS" : "FAILED (using original)"}`);
      return res.json({
        mode: "answer",
        answer: answer,
        match_id: best.id,
        score: best.score,
        enhanced: enhanced
      });
    }

    if (best && best.score >= SUGGESTION_THRESHOLD) {
      return res.json({
        mode: "suggest",
        top: scored.slice(0, TOP_K).map((s) => ({ id: s.id, title: s.title, score: s.score }))
      });
    }

    // If score is very low, question is likely out of scope
    if (!best || best.score < OUT_OF_SCOPE_THRESHOLD) {
      return res.json({
        mode: "fallback",
        reply: "I'm designed to help with questions about IIT Gandhinagar and student counseling services. For other topics, I'd recommend searching online or asking a general assistant. If you have any questions about IIT Gandhinagar, feel free to ask!",
        top: scored.slice(0, TOP_K).map((s) => ({ id: s.id, title: s.title, score: s.score }))
      });
    }

    return res.json({
      mode: "fallback",
      reply: "I couldn't confidently match a FAQ for that. Would you like to rephrase or talk to a counselor?",
      top: scored.slice(0, TOP_K).map((s) => ({ id: s.id, title: s.title, score: s.score }))
    });
  } catch (err) {
    console.error("Error /api/faq-match:", err);
    return res.status(500).json({ error: "server error" });
  }
});

// fetch answer by id
app.post("/api/faq-by-id", (req, res) => {
  const { id } = req.body;
  const f = faqs.find((x) => x.id === id);
  if (!f) return res.status(404).json({ error: "not found" });
  return res.json({ title: f.title, answer: f.a });
});



app.listen(PORT, () => {
  console.log(`Server starting on ${PORT} — initializing model...`);
  init().catch((err) => {
    console.error("Failed to initialize model:", err);
  });
});
