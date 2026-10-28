@'
# AI Customer Ticket Response System

## Overview

A locally running customer support ticket application that combines ticket management, machine-learning-based solution prediction, and retrieval-augmented generation (RAG) to draft detailed support responses from a local knowledge base.

## Features

- Create, view, update, and deactivate support tickets.
- Store ticket details including customer, subject, description, category, priority, status, predicted solution, and detailed response.
- Predict a suggested solution using a locally trained scikit-learn text-classification model.
- Generate detailed responses using a local Ollama language model and retrieved knowledge-base documents.
- Edit and save AI-generated responses.
- Define custom ticket fields, including field types and required settings.
- Store ticket data in a local SQLite database.
- Include sample ticket data and local knowledge-base documents.

## Architecture

1. **Web application and API:** FastAPI, with the user interface served by the application.
2. **Ticket storage:** SQLite.
3. **ML prediction:** TF-IDF text features and Logistic Regression, implemented in `ml_model.py`.
4. **RAG response generation:** `rag_engine.py` retrieves relevant local knowledge-base documents using embeddings and sends the retrieved context to a local Ollama model.
5. **Knowledge base:** Text documents stored in `knowledge_base/`.
6. **Sample data:** `data/sample_tickets.csv`.

## Project Structure

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