from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import gc
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models with device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models with memory optimization
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="/app/cache")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="/app/cache").to(device)
    model.eval()  # Set to evaluation mode

    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", cache_dir="/app/cache")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", cache_dir="/app/cache").to(device)
    detr_model.eval()  # Set to evaluation mode

    # Free up memory
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

def process_prompt(prompt: str) -> str:
    if not prompt:
        return "describe what you see in this image"
    
    prompt = prompt.lower()
    if "what" in prompt and "this" in prompt:
        return prompt
    elif "what" in prompt:
        return f"what is this {prompt.replace('what', '').strip()}"
    else:
        return f"describe this {prompt}"

@app.post("/api/detect")
async def detect_objects(
    prompt: str = Form(None),
    width: int = Form(...),
    height: int = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Read and process image
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large")
            
        image = Image.open(io.BytesIO(contents))
        
        # Resize image if too large
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Object detection
        with torch.no_grad():  # Disable gradient calculation
            inputs = detr_processor(images=image, return_tensors="pt").to(device)
            outputs = detr_model(**inputs)
            
            # Convert outputs to COCO API
            target_sizes = torch.tensor([image.size[::-1]])
            results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            
            detected_objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.9:  # Only include high confidence detections
                    detected_objects.append({
                        "label": detr_model.config.id2label[label.item()],
                        "score": score.item(),
                        "box": box.tolist()
                    })
        
        # Clear memory
        del inputs, outputs, results
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Process prompt
        processed_prompt = process_prompt(prompt)
        
        # Generate caption
        with torch.no_grad():  # Disable gradient calculation
            inputs = processor(image, processed_prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.7,
                do_sample=True
            )
            caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clear memory again
        del inputs, outputs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "message": f"Detected objects: {', '.join(obj['label'] for obj in detected_objects)}. {caption}"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)