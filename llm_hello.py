from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def llm_hello():
    model_dir = "./models"
    # Load the tokenizer and model
    model_name = "gpt2"  # or use a more advanced model like "gpt-3", "gpt-j", etc., if available on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./models")

    # Define the input text
    input_text = "Once upon a time, in a land far away"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2,
                            temperature=0.7, top_k=50, top_p=0.95)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

llm_hello()