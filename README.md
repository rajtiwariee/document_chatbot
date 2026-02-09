# Multi-Tenant Document Chatbot

A production-ready document chatbot with LangGraph ReAct agent for intelligent document retrieval and question answering.

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy 2.0, PostgreSQL
- **AI/ML**: LangGraph, Google Gemini 3, Qdrant
- **Frontend**: React + Vite (TypeScript)
- **Infrastructure**: Docker Compose, GCP Compute Engine

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Google Cloud API Key (for Gemini)

### Setup

1. Clone and configure environment:
```bash
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY
```

2. Start services:
```bash
docker-compose up -d
```

3. Access the application:
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

### Development

For local development without Docker:

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Project Structure

```
personal_document_chatbot/
├── backend/
│   ├── app/
│   │   ├── api/           # API routes
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── agent/         # LangGraph ReAct agent
│   │   ├── document_processing/
│   │   └── vector_store/  # Qdrant integration
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
├── docs/
│   └── ARCHITECTURE.md
├── docker-compose.yml
└── .env.example
```

## License

MIT
