"""
CathodeScreen API - FastAPI backend for cathode materials screening.

Provides endpoints for:
- CIF structure upload and prediction
- Batch predictions
- Model information
"""

from pathlib import Path
from typing import List, Optional
import tempfile
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Security Configuration
API_KEY_NAME = "X-API-Key"
API_KEY = "CATHODE_SCREEN_2026"  # Simple key for demo protection

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )

app = FastAPI(
    title="CathodeScreen API",
    description="AI-powered screening of battery cathode materials",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResult(BaseModel):
    """Single prediction result."""
    material_id: str
    pred_ehull: float
    p_stable: float
    uncertainty: str  # "Low", "Medium", "High"
    action: str  # "DFT", "HOLD", "SKIP"
    confidence_interval: tuple[float, float]


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""
    success: bool
    prediction: Optional[PredictionResult] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    success: bool
    predictions: List[PredictionResult] = []
    n_processed: int = 0
    n_errors: int = 0


class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    model_type: str
    ensemble_size: int
    training_data: str
    daf_at_10: float
    version: str


# In-memory model (will be loaded on startup)
_model = None


def classify_uncertainty(std: float) -> str:
    """Classify uncertainty level."""
    if std < 0.05:
        return "Low"
    elif std < 0.15:
        return "Medium"
    return "High"


def get_action(p_stable: float, unc: str, pred: float) -> str:
    # Relaxed Criteria:
    # 1. Very stable prediction (Ehull < 0.05) with Low uncertainty -> DFT
    # 2. Or Classifier strongly agrees (> 0.7) and hull is decent (< 0.1) -> DFT
    if (unc == "Low" and pred < 0.08) or (p_stable > 0.7 and unc == "Low" and pred < 0.1):
        return "DFT"
    elif p_stable > 0.5 or pred < 0.15:
        return "HOLD"
    return "SKIP"


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "CathodeScreen API", "version": "1.0.0"}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_name="CGCNN-Ensemble",
        model_type="Crystal Graph Convolutional Neural Network",
        ensemble_size=5,
        training_data="Materials Project cathodes (SOAP-LOCO split)",
        daf_at_10=1.64,
        version="1.0.0",
    )


@app.get("/database")
async def get_database(api_key: str = Security(get_api_key)):
    """Get pre-computed predictions from the database."""
    import pandas as pd
    from pathlib import Path
    
    # Try to load predictions file
    predictions_path = Path("data/predictions/ensemble_soap_loco_test.parquet")
    
    if not predictions_path.exists():
        # Raise 500 error instead of failing silently
        raise HTTPException(status_code=500, detail="Predictions file not found on server")
    
    try:
        df = pd.read_parquet(predictions_path)
        
        # Sort by predicted E_hull (best first)
        sort_col = "q50" if "q50" in df.columns else "pred_ehull"
        df = df.sort_values(sort_col, ascending=True)  # Load all materials
        
        # Build response data
        data = []
        for _, row in df.iterrows():
            pred_ehull = row.get("q50", row.get("pred_ehull", 0))
            std = row.get("epistemic_std", 0.1)
            p_stable = row.get("p_stable", 0.5)
            
            # Classify uncertainty
            if std < 0.05:
                unc = "Low"
            elif std < 0.15:
                unc = "Medium"
            else:
                unc = "High"
            
            # Get action
            action = get_action(p_stable, unc, pred_ehull)
            
            data.append({
                "material_id": row.get("material_id", f"mp-{_}"),
                "formula": row.get("formula", ""),
                "pred_ehull": round(float(pred_ehull), 4),
                "p_stable": round(float(p_stable), 3),
                "uncertainty": unc,
                "action": action,
                "confidence_interval": (0.0, 0.0) # Placeholder if needed
            })
        
        return {"success": True, "data": data, "total": len(df)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_structure(cif_file: UploadFile = File(...), api_key: str = Security(get_api_key)):
    """
    Predict stability for a single CIF structure.
    """
    try:
        # Read CIF content
        content = await cif_file.read()
        cif_text = content.decode("utf-8")
        
        from pymatgen.core import Structure
        from .inference import get_predictor
        
        # Parse structure
        structure = Structure.from_str(cif_text, fmt="cif")
        
        # Run real prediction
        predictor = get_predictor()
        result = predictor.predict_structure(structure)
        
        return PredictionResponse(
            success=True,
            prediction=PredictionResult(
                material_id=cif_file.filename or "uploaded",
                pred_ehull=round(result["pred_ehull"], 4),
                p_stable=round(result["p_stable"], 3),
                uncertainty=result["uncertainty"],
                action=result["action"],
                confidence_interval=result["confidence_interval"],
            ),
        )
        
    except Exception as e:
        # CRITICAL: No mock fallback here. Error out.
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(cif_files: List[UploadFile] = File(...), api_key: str = Security(get_api_key)):
    """
    Predict stability for multiple CIF structures.
    
    Upload multiple CIF files for batch processing.
    """
    predictions = []
    n_errors = 0
    
    for cif_file in cif_files:
        try:
            result = await predict_structure(cif_file)
            if result.success and result.prediction:
                predictions.append(result.prediction)
            else:
                n_errors += 1
        except Exception:
            n_errors += 1
    
    return BatchPredictionResponse(
        success=True,
        predictions=predictions,
        n_processed=len(predictions),
        n_errors=n_errors,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
