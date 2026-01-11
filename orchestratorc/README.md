# 📊 Document Analysis Orchestrator

AI-powered legal document processing and case analysis service for JuriAid.

## 🎯 Overview

The Orchestrator is the central hub for document processing in the JuriAid system. It handles file uploads (PDF/TXT), extracts legal content, and performs AI-powered case analysis using Google Gemini.

## ✨ Key Features

- 📄 **Multi-format Support**: PDF and TXT file processing
- 🤖 **AI Analysis**: Google Gemini integration for intelligent case analysis
- 💾 **Knowledge Base**: Persistent storage of analyzed cases
- 🔐 **Secure**: JWT authentication on all protected endpoints
- 📊 **Structured Output**: Organized analysis reports

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
Google Gemini API Key
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

### Run Service

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Visit: http://localhost:8000/docs for API documentation

## 📡 API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | ❌ | Health check |
| `/api/upload-case` | POST | ✅ | Upload & analyze legal case |
| `/api/knowledge-base` | GET | ✅ | Retrieve past analyses |

## 🔧 Configuration

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key
AUTH_SERVICE_URL=http://localhost:8001
ORCHESTRATOR_HOST=127.0.0.1
ORCHESTRATOR_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **AI Model**: Google Gemini (LangChain)
- **File Processing**: PyPDF2
- **Authentication**: JWT

## 📁 Project Structure

```
orchestratorc/
├── app.py                 # Main FastAPI application
├── auth_middleware.py     # JWT verification
├── config.py             # Configuration settings
├── orchestrator/
│   ├── agent_gemini.py   # Gemini AI agent
│   ├── core.py           # Core processing logic
│   └── tools.py          # Utility functions
├── outputs/              # Analysis results
└── uploads/              # Uploaded files
```

## 📝 Usage Example

```python
import requests

# Login to get token
login_response = requests.post(
    "http://localhost:8001/login",
    json={"email": "user@example.com", "password": "password"}
)
token = login_response.json()["access_token"]

# Upload case for analysis
files = {"file": open("case.pdf", "rb")}
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/api/upload-case",
    files=files,
    headers=headers
)

print(response.json())
```

## 🔗 Related Services

- **Auth Service** (Port 8001): User authentication
- **LawStatKG** (Port varies): Law retrieval
- **Past Case Retrieval** (Port 8002): Case search
- **Question Gen** (Port 8003): Legal question generation

---

Part of the **JuriAid Backend System** - See main README for full architecture.