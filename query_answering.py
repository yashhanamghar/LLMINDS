# query_answering.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from google.generativeai import configure, GenerativeModel

# Configure Gemini
configure(api_key="AIzaSyBLWoCLw4b8hrh1qbULhOS-sTWZb2ED9Dc"
)
model_gemini = GenerativeModel("models/gemini-1.5-flash")

# Load FAISS + chunks
faiss_index = faiss.read_index("embeddings/faiss_index/index.faiss")
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_top_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def get_answer(prompt):
    response = model_gemini.generate_content(prompt)
    return response.text.strip()
    
    response = model_gemini.generate_content(prompt)
    return response.text

