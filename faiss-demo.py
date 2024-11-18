from transformers import AutoTokenizer, AutoModel, pipeline
import faiss
import numpy as np
import torch  # Import torch here


# Initialize embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, cache_dir="./models")
embedding_model = AutoModel.from_pretrained(embedding_model_name, cache_dir="./models")

# Initialize text generator pipeline (using GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Sample corpus
documents = [
    "Python is a popular programming language for AI and machine learning.",
    "FAISS is a library for efficient similarity search, often used in AI.",
    "RAG combines retrieval systems with generative models to enhance responses.",
    "GPT models can generate human-like text for a variety of applications.",
    "Vector databases store embeddings to enable fast similarity searches."
]

# Function to create embeddings
def get_embeddings(texts):
    """
    Generate embeddings for a list of texts using the MiniLM model.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()

# Create FAISS index
def create_faiss_index(documents):
    """Index the documents using FAISS."""
    document_embeddings = get_embeddings(documents)
    document_vectors = np.array(document_embeddings, dtype="float32")
    dimension = document_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_vectors)
    return index

# Initialize FAISS index
index = create_faiss_index(documents)

# Search FAISS index
def search_faiss(query, top_k=2):
    """Retrieve the most relevant documents using FAISS."""
    query_embedding = get_embeddings([query])[0].reshape(1, -1).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [documents[i] for i in indices[0]]
    return results

# Generate response
def generate_response(query):
    """Retrieve relevant documents and generate a response using GPT-2."""
    # Retrieve relevant documents
    relevant_docs = search_faiss(query)
    context = "\n".join(relevant_docs)

    # Construct prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate text using the pipeline
    response = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7)[0]['generated_text']
    return response

# Test the pipeline
if __name__ == "__main__":
    user_query = "What is FAISS used for?"
    answer = generate_response(user_query)
    print(f"Query: {user_query}\nAnswer: {answer}")
