from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Initialize PaddleOCR (CPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Run OCR
    result = ocr.ocr(image_np)

    # Prepare structured output
    extracted = []
    for line in result:
        for word_info in line:
            box = word_info[0]  # bounding box coordinates
            text, confidence = word_info[1]
            # Convert box to dictionary with xmin, ymin, xmax, ymax
            xmin = int(min([p[0] for p in box]))
            ymin = int(min([p[1] for p in box]))
            xmax = int(max([p[0] for p in box]))
            ymax = int(max([p[1] for p in box]))
            extracted.append({
                "text": text,
                "confidence": float(confidence),
                "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            })

    return {"filename": file.filename, "results": extracted}