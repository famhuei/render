from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import base64

app = FastAPI()

# Enable CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]
)

# Load model and processor for better image understanding
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Load object detection model
object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

class CoordinatesModel(BaseModel):
    response: str = None
    error: str = None

def process_prompt(prompt: str) -> str:
    """Process the user prompt to get better results"""
    prompt = prompt.lower()
    if "what is this" in prompt or "what do you see" in prompt:
        return "Describe this object in detail"
    elif "describe" in prompt:
        return "Describe this object in detail"
    return prompt

@app.post("/api/detect")
async def detect_objects(
    prompt: str = Form(...),
    width: str = Form(...),
    height: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Read and process image
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        
        # First detect objects in the image
        detections = object_detector(image)
        detected_objects = [det['label'] for det in detections]
        
        # Process the prompt
        processed_prompt = process_prompt(prompt)
        
        # Generate caption
        inputs = processor(image, text=processed_prompt, return_tensors="pt")
        generated_ids = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            temperature=0.7
        )
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Combine object detection with caption
        if detected_objects:
            response = f"I can see {', '.join(detected_objects)}. {caption}"
        else:
            response = caption
            
        return CoordinatesModel(response=response)
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return CoordinatesModel(error=str(e))

if __name__ == "__main__":
    # Use 0.0.0.0 to allow external connections
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 