# different types of encodings: https://www.youtube.com/watch?v=wgfSDrqYMJ4
from sentence_transformers import SentenceTransformer, util

def simple_sentence_transformer():
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="./models")
    embeddings = model.encode(sentences)
    print(embeddings)

def transformer_based_embedding():
    # Load a transformer-based model
    model = SentenceTransformer('all-MiniLM-L6-v2',cache_folder="./models")

    # Define two sentences
    sentence1 = "The Eiffel Tower is in Paris."
    sentence2 = "Paris is home to the Eiffel Tower."

    # Generate embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    # Compute cosine similarity between embeddings
    cosine_sim = util.pytorch_cos_sim(embedding1, embedding2)
    print(f"Cosine Similarity: {cosine_sim.item():.4f}")

transformer_based_embedding()