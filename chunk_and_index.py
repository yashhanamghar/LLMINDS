import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

def load_documents(folder_path):
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            with open(os.path.join(folder_path, fname), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    docs = load_documents('data/pdf_text')
    
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    index, embeddings = build_faiss_index(all_chunks, model)
    
    # Save index and chunks
    faiss.write_index(index, 'embeddings/faiss_index/index.faiss')
    with open('data/chunks.pkl', 'wb') as f:
        pickle.dump(all_chunks, f)
