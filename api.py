"""FastAPI web interface for AI Outlier Detection Pipeline."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import tempfile
import pandas as pd
import json
import os
from src.pipeline import AIOutlierDetectionPipeline

app = FastAPI(
    title="AI Outlier Detection API",
    description="Semantic anomaly detection using embeddings and machine learning",
    version="1.0.0"
)

# Global pipeline instance
pipeline = None

class TextAnalysisRequest(BaseModel):
    texts: List[str]
    categories: Optional[List[str]] = None

class OutlierResponse(BaseModel):
    method: str
    outlier_count: int
    outlier_indices: List[int]
    outlier_texts: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    global pipeline
    try:
        pipeline = AIOutlierDetectionPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")

@app.get("/")
async def root():
    """API health check."""
    return {"message": "AI Outlier Detection API", "status": "running"}

@app.post("/analyze/texts", response_model=Dict[str, OutlierResponse])
async def analyze_texts(request: TextAnalysisRequest):
    """Analyze outliers in provided texts."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        results = pipeline.detect_outliers_in_text(
            request.texts, 
            request.categories
        )
        
        response = {}
        for method, data in results.items():
            response[method] = OutlierResponse(
                method=method,
                outlier_count=data['outlier_count'],
                outlier_indices=data['outlier_indices'],
                outlier_texts=data['outlier_texts']
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    text_column: str = "text",
    category_column: Optional[str] = None
):
    """Analyze outliers in uploaded CSV/JSON file."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif file.filename.endswith('.json'):
            df = pd.read_json(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found")
        
        texts = df[text_column].tolist()
        categories = df[category_column].tolist() if category_column and category_column in df.columns else None
        
        results = pipeline.detect_outliers_in_text(texts, categories)
        
        return {
            "file_info": {
                "filename": file.filename,
                "total_documents": len(texts),
                "text_column": text_column,
                "category_column": category_column
            },
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@app.post("/pipeline/run")
async def run_full_pipeline():
    """Run the complete outlier detection pipeline on 20 Newsgroups data."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        results = pipeline.run_full_pipeline(save_results=True, output_dir="api_results")
        
        return {
            "status": "completed",
            "total_documents": results['total_documents'],
            "summary_stats": results['summary_stats'],
            "output_directory": "api_results"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Download result files."""
    file_path = os.path.join("api_results", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.get("/results")
async def list_results():
    """List available result files."""
    results_dir = "api_results"
    
    if not os.path.exists(results_dir):
        return {"files": []}
    
    files = []
    for filename in os.listdir(results_dir):
        file_path = os.path.join(results_dir, filename)
        if os.path.isfile(file_path):
            files.append({
                "filename": filename,
                "size": os.path.getsize(file_path)
            })
    
    return {"files": files}

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", 1))
    
    if os.getenv("ENVIRONMENT") == "production":
        # Production should use gunicorn, this is fallback
        uvicorn.run(app, host=host, port=port, workers=workers)
    else:
        # Development mode
        uvicorn.run(app, host=host, port=port, reload=True)