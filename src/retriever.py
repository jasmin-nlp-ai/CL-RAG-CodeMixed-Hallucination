import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading BGE-M3 model...")

model = SentenceTransformer("BAAI/bge-m3")

with open("data/passages.json", "r", encoding="utf-8") as f:
    passages = json.load(f)

documents = [item["text"] for item in passages]

print("Creating document embeddings...")
doc_embeddings = model.encode(documents)

query = input("\nEnter your query: ")

query_embedding = model.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

best_idx = scores.argmax()

print("\nTop Result:")
print(documents[best_idx])

print(f"\nSimilarity Score: {scores[best_idx]:.4f}")