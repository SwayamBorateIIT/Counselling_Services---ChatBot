import fs from "fs";
import { pipeline } from "@xenova/transformers";

/*
  This script:
  1. Reads faqs.json
  2. Generates embeddings for each FAQ question
  3. Saves faqs_with_embeddings.json

  Run:
  node embeddings.js
*/

const FAQ_INPUT_FILE = "./faq.json";
const FAQ_OUTPUT_FILE = "./faqs_with_embeddings.json";
const EMBEDDING_MODEL = "Xenova/bge-small-en-v1.5";

async function generateEmbeddings() {
  console.log("Loading FAQ data...");
  const faqs = JSON.parse(fs.readFileSync(FAQ_INPUT_FILE, "utf-8"));

  console.log("Loading embedding model (first time may take a minute)...");
  const embedder = await pipeline(
    "feature-extraction",
    EMBEDDING_MODEL
  );

  console.log("Generating embeddings...");
  const faqsWithEmbeddings = [];

  for (let i = 0; i < faqs.length; i++) {
    const faq = faqs[i];

    // Generate embedding for the question (best practice)
    const embedding = await embedder(faq.question, {
      pooling: "mean",
      normalize: true
    });

    faqsWithEmbeddings.push({
      id: i + 1,
      question: faq.question,
      answer: faq.answer,
      embedding: Array.from(embedding.data)
    });

    console.log(`Embedded FAQ ${i + 1}/${faqs.length}`);
  }

  fs.writeFileSync(
    FAQ_OUTPUT_FILE,
    JSON.stringify(faqsWithEmbeddings, null, 2)
  );

  console.log(`Done! Embeddings saved to ${FAQ_OUTPUT_FILE}`);
}

generateEmbeddings().catch(err => {
  console.error("Error generating embeddings:", err);
});
