// embeddingWorker.js - Piscina compatible
import { pipeline } from '@xenova/transformers';

let embedder = null;

export default async function (text) {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
    console.log('[Worker] Embedding model loaded');
  }
  
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}