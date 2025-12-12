// precompute_embeddings.js
// Usage: node precompute_embeddings.js
// Produces faqs_with_embeddings.json from faqs_raw.json using local BGE-small via @xenova/transformers

import fs from "fs";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";

dotenv.config();

const RAW = "./faqs_raw.json";
const OUT = "./faqs_with_embeddings.json";
const MODEL = "Xenova/bge-small-en-v1.5"; // BGE-small

async function main() {
  // load raw FAQs
  if (!fs.existsSync(RAW)) {
    console.error("faqs_raw.json not found in project root.");
    process.exit(1);
  }
  const raw = JSON.parse(fs.readFileSync(RAW, "utf8"));

  console.log("Loading embedding model (this will download model files on first run)...");
  const extractor = await pipeline("feature-extraction", MODEL);

  const output = [];
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i];
    console.log(`Embedding ${i + 1}/${raw.length} → ${item.title || item.q}`);
    // pool to a single vector: many pipelines return nested arrays; we take mean pooling
    const embOut = await extractor(item.q, { pooling: "mean", normalize: true });
    // embOut.data is typically the vector (depends on pipeline output shape)
    const vector = embOut.data ?? embOut; // fallback
    output.push({ ...item, embedding: vector });
    // small pause (safeguard)
    await new Promise((r) => setTimeout(r, 50));
  }

  fs.writeFileSync(OUT, JSON.stringify(output, null, 2));
  console.log("Done! Wrote", OUT);
  process.exit(0);
}

main().catch((err) => {
  console.error("Error in precompute_embeddings:", err);
  process.exit(1);
});
