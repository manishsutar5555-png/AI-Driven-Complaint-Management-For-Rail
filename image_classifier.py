# models/image_classifier.py
import os
from PIL import Image
from io import BytesIO
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration

# Using BLIP model to generate caption from image
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def classify_image(image_path: str, topk: int = 1):
    """
    Convert image to descriptive text using BLIP (LLM-based captioning).
    Returns list of (text, score) with score=1.0 (confidence placeholder)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs)
        text = processor.decode(out[0], skip_special_tokens=True)
        return [(text, 1.0)]
    except Exception as e:
        print("Image classification failed:", e)
        return [("unable_to_describe_image", 1.0)]
