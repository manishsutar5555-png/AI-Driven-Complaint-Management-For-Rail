from transformers import MarianMTModel, MarianTokenizer

# Load multilingual → English model
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_to_english(text: str) -> str:
    """Translate any input text into English."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text  # fallback to original

