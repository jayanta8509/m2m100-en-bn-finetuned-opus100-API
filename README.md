# English-Bengali Translation API

A FastAPI-based translation service that translates between English and Bengali using the M2M100 (Many-to-Many) model fine-tuned on OPUS100 dataset.

## 🌟 Features

- **Bidirectional Translation**: English ↔ Bengali
- **FastAPI Framework**: High-performance, modern web framework
- **M2M100 Model**: State-of-the-art multilingual translation model
- **GPU Support**: Automatic CUDA detection and usage
- **CORS Enabled**: Cross-origin requests supported
- **Health Monitoring**: Built-in health check endpoint
- **Interactive Documentation**: Auto-generated API docs
- **Error Handling**: Comprehensive error handling and validation

## 📹 Video Demo

Watch the API in action! This video demonstrates the complete setup and usage of the English-Bengali Translation API.

https://github.com/user-attachments/assets/transition.mp4

**Video Contents:**
- 🚀 API setup and installation
- 🔧 Model loading and configuration
- 🌐 API endpoint testing
- 📱 Frontend integration example
- 🔍 Error handling demonstration
- 📊 Performance monitoring

> **Note**: Make sure the `transition.mp4` file is in your repository root directory for the video to display properly.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd m2m100-en-bn-finetuned-opus100-API
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   ```

3. **Activate virtual environment**
   
   **Windows:**
   ```bash
   env\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   source env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

1. **Start the server**
   ```bash
   python app.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. **Access the API**
   - **API Base URL**: `http://localhost:8000`
   - **Interactive Docs**: `http://localhost:8000/docs`
   - **Health Check**: `http://localhost:8000/health`

## 📚 API Endpoints

### 1. Health Check
```http
GET /health
```
Returns the status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. Translation
```http
POST /translate
```
Translate text between English and Bengali.

**Request Body:**
```json
{
  "text": "Hello, how are you?",
  "source_lang": "en",
  "target_lang": "bn"
}
```

**Response:**
```json
{
  "original_text": "Hello, how are you?",
  "translated_text": "হ্যালো, আপনি কেমন আছেন?",
  "source_language": "en",
  "target_language": "bn",
  "success": true,
  "message": "Translation completed successfully"
}
```

### 3. Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

## 🔧 Usage Examples

### Using cURL

**English to Bengali:**
```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Good morning, have a great day!",
       "source_lang": "en",
       "target_lang": "bn"
     }'
```

**Bengali to English:**
```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "আপনি কেমন আছেন?",
       "source_lang": "bn",
       "target_lang": "en"
     }'
```

### Using Python

```python
import requests

# API endpoint
url = "http://localhost:8000/translate"

# Translation data
data = {
    "text": "Hello, how are you today?",
    "source_lang": "en",
    "target_lang": "bn"
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(f"Original: {result['original_text']}")
print(f"Translated: {result['translated_text']}")
```

### Using JavaScript (Frontend)

```javascript
async function translateText(text, sourceLang, targetLang) {
    const response = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            source_lang: sourceLang,
            target_lang: targetLang
        })
    });
    
    const result = await response.json();
    return result.translated_text;
}

// Usage
translateText("Hello world", "en", "bn")
    .then(translation => console.log(translation));
```

## 📁 Project Structure

```
m2m100-en-bn-finetuned-opus100-API/
├── app.py                 # FastAPI application and endpoints
├── model.py              # M2M100 model loading and translation logic
├── requirements.txt      # Python dependencies
├── test.py              # Original test script
├── README.md            # This file
└── env/                 # Virtual environment
```

## 🛠️ Configuration

### Model Configuration
The model is configured in `model.py`:
- **Model Name**: `Jayanta8509/m2m100-en-bn-finetuned-opus100`
- **Max Length**: 128 tokens
- **Beam Search**: 5 beams
- **Device**: Auto-detection (CUDA/CPU)

### API Configuration
The API is configured in `app.py`:
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `8000`
- **CORS**: Enabled for all origins
- **Max Text Length**: 1000 characters

## 📦 Dependencies

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **transformers**: Hugging Face transformers library
- **torch**: PyTorch deep learning framework
- **pydantic**: Data validation
- **python-multipart**: Form data handling

## 🔍 Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input or unsupported language pairs
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Translation processing errors

## 🚀 Deployment

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t translation-api .
docker run -p 8000:8000 translation-api
```

### Production Considerations

1. **Security**: Restrict CORS origins in production
2. **Rate Limiting**: Implement rate limiting for API endpoints
3. **Monitoring**: Add logging and monitoring
4. **Scaling**: Use multiple workers for high traffic
5. **Caching**: Implement response caching for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library
- **Facebook AI**: For the M2M100 model
- **OPUS**: For the parallel corpus dataset
- **Jayanta8509**: For the fine-tuned model

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the health endpoint at `/health`

---

**Made with ❤️ for English-Bengali translation**
