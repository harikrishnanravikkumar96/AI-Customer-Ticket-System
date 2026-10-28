# AI Customer Ticket Response System

## 1. Overview

A local customer support ticket application combining ticket management, machine-learning-based solution prediction, and Retrieval-Augmented Generation (RAG) to draft detailed responses grounded in a local knowledge base.

## 2. Features

- Create, read, update, and deactivate support tickets.
- Predict suggested solutions using a scikit-learn text-classification model.
- Generate detailed responses using a local Ollama language model and retrieved knowledge-base documents.
- Edit and save AI-generated responses.
- Configure custom ticket fields with text, dropdown, number, date, and checkbox types.
- Store tickets and custom-field definitions in SQLite.
- Include sample training data and knowledge-base documents.

## 3. Architecture

1. **Frontend and API:** FastAPI serves the web interface and API endpoints.
2. **Database:** SQLite stores tickets and configurable field definitions.
3. **ML prediction:** TF-IDF text features and Logistic Regression predict a solution category. The application maps that category to a suggested solution.
4. **Embeddings and retrieval:** The RAG module creates embeddings for local knowledge documents and retrieves relevant documents using similarity scoring.
5. **Response generation:** The retrieved context is passed to the local Ollama language model to generate a detailed response.
6. **Human review:** Users can edit and save the generated response.

## 4. Project Structure

```text
AI-Customer-Ticket-System/
├── main.py
├── ml_model.py
├── rag_engine.py
├── solution_model.joblib
├── knowledge_embeddings.json
├── requirements.txt
├── data/
│   └── sample_tickets.csv
└── knowledge_base/
    ├── login_troubleshooting.txt
    ├── order_tracking.txt
    ├── password_reset.txt
    └── payment_failure.txt
```

## 5. Prerequisites

- Python 3.11 or later
- Ollama installed and running locally
- The configured local models:
  - `llama3.2:3b` for response generation
  - `nomic-embed-text` for embeddings

Download the models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## 6. Installation

Open PowerShell in the project directory.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 7. Database Setup

The application uses SQLite and stores its local database in `tickets.db`.

On application startup, `main.py` calls the database initialization function. It creates the following tables if they do not already exist:

- **tickets:** Stores ticket ID, customer name, subject, description, category, priority, predicted solution, detailed response, status, dynamic field values, creation and update timestamps, and active status.
- **field_definitions:** Stores custom field names, labels, types, required settings, and dropdown options.

No separate database server or manual schema creation is required. The database file is created locally when the application initializes. It is excluded from version control.

## 8. Run the Application

Ensure Ollama is running and the required models are available. Then run:

```powershell
uvicorn main:app --reload
```

Open the web interface:

http://127.0.0.1:8000

Keep the terminal open while using the application.

## 9. ML Prediction Workflow

1. The user enters a ticket subject and description.
2. The ML pipeline converts the text into TF-IDF features.
3. A trained Logistic Regression classifier predicts a category.
4. The predicted category maps to a predefined suggested solution.
5. The suggested solution is stored with the ticket.

The model can be retrained using the training examples defined in `ml_model.py`:

```powershell
python ml_model.py
```

The training script generates the model artifact used by the application. Prediction quality depends on the quantity, coverage, and quality of the training examples.

## 10. LLM and RAG Workflow

1. The user requests a detailed response for a ticket.
2. The RAG module compares the ticket content with the local knowledge-base documents using embeddings.
3. Relevant documents are retrieved and supplied as context.
4. The local Ollama language model generates a detailed response.
5. The response and retrieved source names are returned to the application.
6. The generated response is saved and can be edited by the user.

Knowledge-base documents are stored in `knowledge_base/`. Add or update these documents to change the available support information.

AI inference and embedding generation use local Ollama models; the documented workflow does not require a cloud inference API.

## 11. Configuration and Data

- `main.py`: FastAPI application, user interface, ticket operations, and database initialization.
- `ml_model.py`: ML training examples, model training, category mapping, and predictions.
- `rag_engine.py`: Knowledge retrieval, embeddings, and local LLM response generation.
- `data/sample_tickets.csv`: Sample ticket data.
- `knowledge_base/`: Sample support knowledge documents.
- `solution_model.joblib`: Trained ML model artifact.
- `knowledge_embeddings.json`: Cached knowledge-document embeddings.
- `requirements.txt`: Python dependencies.

## 12. Known Limitations

- ML predictions depend on the available training examples and should be reviewed by a support agent.
- RAG response quality depends on the relevance and completeness of the knowledge base.
- Detailed response generation requires the configured Ollama service and models to be available.
- This is an assessment prototype; generated responses should be reviewed before being sent to customers.
