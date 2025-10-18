from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from model import translation_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="English-Bengali Translation API",
    description="API for translating between English and Bengali using m2m100-en-bn-finetuned-opus100 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response
class TranslationRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "en"
    target_lang: Optional[str] = "bn"

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    success: bool
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    logger.info("Starting up the application...")
    success = translation_model.load_model()
    if success:
        logger.info("Application started successfully with model loaded")
    else:
        logger.error("Failed to load model during startup")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are working properly"""
    return HealthResponse(
        status="healthy" if translation_model.model is not None else "unhealthy",
        model_loaded=translation_model.model is not None,
        device=translation_model.device
    )

# Main translation endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between English and Bengali
    
    - **text**: The text to translate
    - **source_lang**: Source language (en or bn)
    - **target_lang**: Target language (en or bn)
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long. Maximum 1000 characters allowed.")
        
        # Check if model is loaded
        if translation_model.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
        
        # Perform translation based on language pair
        if request.source_lang == "en" and request.target_lang == "bn":
            translated_text = translation_model.translate_en_to_bn(request.text)
        elif request.source_lang == "bn" and request.target_lang == "en":
            translated_text = translation_model.translate_bn_to_en(request.text)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported language pair. Only 'en' to 'bn' and 'bn' to 'en' are supported."
            )
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_lang,
            target_language=request.target_lang,
            success=True,
            message="Translation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "English-Bengali Translation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "translate": "/translate",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
