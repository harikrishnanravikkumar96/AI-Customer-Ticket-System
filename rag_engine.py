
import json
import math
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
CACHE_PATH = BASE_DIR / "knowledge_embeddings.json"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

SAMPLE_DOCS = {
    "password_reset.txt": """
Password Reset Guide

If a customer forgets their password, direct them to the
Forgot Password option on the sign-in page. They should
enter their registered email address and follow the reset
instructions received from the service. They should check
their spam folder if the email does not arrive. Never ask
the customer to share their password or verification code.
If the reset email still does not arrive, escalate the issue
to the support team.
""",
    "payment_failure.txt": """
Payment Failure Guide

If a payment fails, first check the transaction status in
the payment system. If money was deducted but the order
was not confirmed, do not ask the customer to pay again
immediately. Verify the transaction and follow the official
refund or reconciliation process. Escalate unresolved
transactions to the payment support team. Do not promise
a refund date unless it has been confirmed.
""",
    "order_tracking.txt": """
Order Tracking Guide

For a delayed order, verify the order number and check the
latest tracking information. Explain the available delivery
status to the customer. If tracking has not updated or the
package is overdue, raise a request with the delivery team.
Do not invent a delivery date that is not confirmed.
""",
    "login_troubleshooting.txt": """
Login Troubleshooting Guide

If a customer cannot log in, check whether the registered
email address is correct and whether the account is locked.
If the customer forgot the password, follow the password
reset guide. If the problem continues after resetting the
password, collect the error message and escalate to technical
support. Never request passwords or one-time verification codes.
""",
}


def ensure_knowledge_base():
    KB_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        path = KB_DIR / filename
        if not path.exists():
            path.write_text(content.strip(), encoding="utf-8")


def embed_texts(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def load_knowledge():
    ensure_knowledge_base()

    documents = []
    for path in sorted(KB_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({
                "source": path.name,
                "text": content,
            })

    if not documents:
        raise RuntimeError("Knowledge base contains no text documents.")

    # Reuse saved embeddings if the document contents have not changed.
    fingerprint = [
        {"source": d["source"], "text": d["text"]}
        for d in documents
    ]

    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("documents") == fingerprint:
                if len(cached.get("embeddings", [])) == len(documents):
                    return documents, cached["embeddings"]
        except (ValueError, OSError):
            pass

    embeddings = embed_texts([d["text"] for d in documents])

    CACHE_PATH.write_text(
        json.dumps({
            "documents": fingerprint,
            "embeddings": embeddings,
        }),
        encoding="utf-8",
    )

    return documents, embeddings


def retrieve_context(query, top_k=3):
    documents, embeddings = load_knowledge()
    query_embedding = embed_texts([query])[0]

    ranked = [
        (
            cosine_similarity(query_embedding, embedding),
            document,
        )
        for document, embedding in zip(documents, embeddings)
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "source": document["source"],
            "text": document["text"],
            "score": round(score, 4),
        }
        for score, document in ranked[:top_k]
    ]


def generate_response(subject, description):
    query = f"{subject}\n{description}"
    retrieved = retrieve_context(query)

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in retrieved
    )

    prompt = f"""
You are a friendly customer support assistant.

Answer the customer's issue using the supplied knowledge base.
Do not invent policies, transaction results, or delivery dates.
If the knowledge base does not contain enough information,
say that the support team needs to investigate.
Never ask for passwords or one-time verification codes.
Keep the answer clear, helpful, and professional.

KNOWLEDGE BASE:
{context}

CUSTOMER TICKET:
Subject: {subject}
Description: {description}

Write the customer-facing response:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()

    if not answer:
        raise RuntimeError("The local LLM returned an empty response.")

    return {
        "response": answer,
        "sources": [
            {
                "source": item["source"],
                "score": item["score"],
            }
            for item in retrieved
        ],
    }


if __name__ == "__main__":
    print("Preparing local knowledge base...")
    docs, embeddings = load_knowledge()
    print(f"Indexed {len(docs)} knowledge-base documents.")

    result = generate_response(
        "Forgot password",
        "I cannot sign in because I forgot my password. "
        "The reset email has not arrived.",
    )

    print("\nRetrieved sources:")
    for source in result["sources"]:
        print(f"- {source['source']} (similarity: {source['score']})")

    print("\nGenerated response:")
    print(result["response"])
