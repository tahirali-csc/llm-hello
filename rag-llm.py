
# Import necessary libraries
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.neighbors import NearestNeighbors
import torch

# Step 1: Prepare Custom Data
documents = [
    {"id": "1", "text": "The Eiffel Tower is located in Paris, France."},
    {"id": "2", "text": "The Great Wall of China is one of the wonders of the world."},
    {"id": "3", "text": "Mount Everest is the highest mountain in the world, located in the Himalayas."},
    {"id": "4", "text": "The Amazon Rainforest is the largest tropical rainforest in the world."},
    {"id": "5", "text": "The Leaning Tower of Pisa is a famous landmark in Italy."}
    # Add more documents as needed
]

# Step 2: Load Sentence Embedding Model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,cache_dir="./models")
embedding_model = AutoModel.from_pretrained(embedding_model_name,cache_dir="./models")

# Function to embed texts
def embed_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Embed all documents
texts = [doc['text'] for doc in documents]
document_embeddings = embed_texts(texts)

# Step 3: Set Up Nearest Neighbors Model
knn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(document_embeddings)

# Step 4: Define Retrieval Function
def retrieve(query, k=3):
    query_embedding = embed_texts([query])
    distances, indices = knn.kneighbors(query_embedding, n_neighbors=k)
    results = [documents[i]['text'] for i in indices[0]]
    return results

# Step 5: Load Text Generation Model (e.g., GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Step 6: Define RAG Pipeline Function
def rag_pipeline(query):
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)

    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    answer = generator(input_text, max_length=100, num_return_sequences=1)[0]["generated_text"]

    answer_text = answer.split("Answer:")[-1].strip()
    return answer_text

# Test the RAG Pipeline
query = "Where is the Eiffel Tower located?"
print(f"Query: {query}")
print("Answer:", rag_pipeline(query))
