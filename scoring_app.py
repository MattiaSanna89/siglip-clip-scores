import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Union
from PIL import Image
import torch
import numpy as np
from io import BytesIO
from scoring import MultimodalSimilarity  # Import the class

app = FastAPI()

# Load model and processor
mms = MultimodalSimilarity()

class ScoreResponse(BaseModel):
    siglipProbPool: Union[float, List]
    siglipProbSeq: Union[float, List]
    clipScorePool: Union[float, List]
    clipScoreSeq: Union[float, List]

class Tokenizers(BaseModel):
    siglipTokens: Union[List, int]
    clipTokens: Union[List, int]

@app.post("/count_tokens", response_model=Tokenizers)
def calculate_tokens(texts: List[str] = Form(...)):
    siglip_tokens = [len(toks) for toks in mms.siglip_processor.tokenizer(texts)['input_ids']]
    clip_tokens = [len(toks) for toks in mms.siglip_processor.tokenizer(texts)['input_ids']]
    return {"siglipTokens": siglip_tokens, "clipTokens": clip_tokens}

@app.post("/scores", response_model=ScoreResponse)
async def score_images(texts: List[str] = Form(...), files: List[UploadFile] = File(...)):
    # Load all images
    images = []
    for file in files:
        img_data = await file.read()
        image = Image.open(BytesIO(img_data)).convert("RGB")
        images.append(image)

    siglip_output = mms.siglip_proba(texts, images)
    clip_output = mms.clip_similarity(texts, images)

    print(siglip_output, clip_output)
    
    return {
        "siglipProbPool": siglip_output['pooling_score']['probabilities'].tolist(), 
        "siglipProbSeq": siglip_output['sequence_score']['probabilities'].tolist(),
        "clipScorePool": clip_output['pooling_score'].tolist(),
        "clipScoreSeq": clip_output['sequence_score'].tolist()
    }

# Run the server when the script is executed
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)