# AkadVerse: Marketplace Recommender

**Tier 3 Algorithm / Classical ML | Microservice Port: `8001`**

Drives the campus economy by providing personalized product and service recommendations using collaborative filtering and real-time sentiment analysis.

## Table of Contents
- [What This Microservice Does](#what-this-microservice-does)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Getting Your API Key](#getting-your-api-key)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Testing with Swagger UI](#testing-with-swagger-ui)
- [Example Test Inputs](#example-test-inputs)
- [Understanding the Responses](#understanding-the-responses)
- [Generated Files](#generated-files)
- [Common Errors and Fixes](#common-errors-and-fixes)
- [Project Structure](#project-structure)

## What This Microservice Does

This service is a Tier 3 component of the AkadVerse AI-first e-learning platform. It lives within the Platform Core module and serves Students and Student Entrepreneurs.

**Core Workflow:**

1.  **Event Consumption:** Listens for `order.completed`, `click.event`, and `business.registered` events via the simulated Kafka webhook.
2.  **Matrix Factorization:** Uses the SVD (Singular Value Decomposition) algorithm to identify latent factors in student purchasing behavior.
3.  **Hybrid Ranking:** Combines collaborative filtering predictions with aggregate campus sentiment scores for a weighted recommendation list.
4.  **Status Tracking:** Cross-references user history to label items as "Purchased" or "Available" in real-time.

**Key Design Decisions:**

*   **Strategic Retraining:** To ensure scalability, the model only retrains on high-intent events (orders/registrations) and ignores low-intent clicks.
*   **Sentiment Boosting:** A 20 percent ranking boost is applied to items with positive campus-wide sentiment.
*   **Upsert Sync:** Results are persisted to MongoDB Atlas using an upsert strategy to ensure a single, fresh record per student.

## Architecture Overview

```
Platform Events (Kafka)
    |
    v
[FastAPI Webhook] (Python 3.12)
    |
    v
[SVD Algorithm] (Scikit-Surprise) <--- [Mock Dataset] (Pandas)
    |
    v
[Hybrid Logic] (Sentiment + Ranking)
    |
    v
[MongoDB Atlas] (Marketplace Recommendations Collection)
```

| Component      | Technology             | Purpose                      |
| :------------- | :--------------------- | :--------------------------- |
| API Layer      | FastAPI                | Async REST endpoints         |
| AI Model       | Scikit-Surprise (SVD)  | Collaborative filtering      |
| Data Processing| Pandas                 | Matrix manipulation          |
| Cloud Storage  | MongoDB Atlas          | Persistence for the Insight Engine |

## Prerequisites

*   Python 3.10 or higher
*   `pip`
*   MongoDB Atlas account and connection string
*   Basic knowledge of FastAPI and Uvicorn

## Getting Your API Key

For the Marketplace Recommender, the primary credential required is your MongoDB URI.

1.  Log in to your MongoDB Atlas dashboard.
2.  Navigate to Database -> Connect.
3.  Choose Drivers and select Python.
4.  Copy the connection string (the MONGO_URI).
5.  Replace `<password>` with your actual database password.

## Installation

**Step 1 -- Set up your project folder**

Create dedicated folder: `akadverse-marketplace-recommender/`

**Step 2 -- Create and activate a virtual environment**

*   **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
*   **macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**Step 3 -- Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 -- Configure environment variables**

Create a `.env` file in the root directory:

```env
MONGO_URI=your_mongodb_connection_string_here
```

## Running the Server

```bash
uvicorn marketplace_api:app --host 127.0.0.1 --port 8001 --reload
```

Expected terminal output:

```
Refreshing AkadVerse marketplace AI model...
Model training complete.
[INFO] Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

## API Endpoints

### 1. `POST /webhook/event`

**What it does:** Receives platform events and triggers background tasks for retraining or data updates.

**Success response (200 OK):**

```json
{
  "status": "success",
  "message": "Order queued for processing."
}
```

### 2. `GET /predict-interest`

**What it does:** Returns a predicted rating (1-5) for a specific student and item.

### 3. `GET /top-recommendations`

**What it does:** Generates a personalized top-5 list with status labels and sentiment boosting.

## Testing with Swagger UI

With the server running, open:
[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

## Example Test Inputs

**Test 1 -- Check Recommendations**

`GET /top-recommendations?student_id=21EE044999`

Expected: List of items with "Purchased" or "Available" status.

**Test 2 -- High Sentiment Boost**

`POST /webhook/event` with:

```json
{
  "event_type": "order.completed",
  "student_id": "19MC022145",
  "payload": {
    "item_id": "Acoustic Guitar Capo",
    "rating": 5.0,
    "sentiment": 1.0
  }
}
```

Expected: Capo score increases in subsequent GET requests.

## Understanding the Responses

 `"status": "Purchased"` flag**
    This is a reactive label. It indicates that the student's ID was found in the interaction matrix for that specific item.

*   **Why does the model retrain?**
    Collaborative filtering requires the entire matrix to be recalculated to find new latent patterns when a new purchase is made.

## Generated Files

| File / Folder        | What it is              | Gitignore?              |
| :------------------- | :---------------------- | :---------------------- |
| `.env`               | MongoDB credentials     | Yes -- never commit     |
| `venv/`              | Virtual environment     | Yes                     |
| `__pycache__/`       | Python byte code        | Yes                     |

## Common Errors and Fixes

*   **Error: `MONGO_URI` environment variable not set**
    Ensure your `.env` file is present and the variable name matches exactly.

*   **HTTPException: 503 Recommendation engine is offline**
    This occurs if the initial model training fails. Check the terminal for Pandas or Surprise library errors.

## Project Structure

```
akadverse-marketplace-recommender/
|-- marketplace_api.py       # Main microservice logic
|-- requirements.txt         # Dependencies
|-- .env                     # Hidden secrets
|-- .gitignore               # Excludes environment files
|-- README.md                # This file
```

## Part of the AkadVerse Platform

This microservice is Tier 3 in the AkadVerse AI architecture, operating within the Platform Core alongside:

*   YouTube Recommender (Port 8000)
*   Insight Engine (Port 8010)

The marketplace.recommendation data synced to MongoDB is used by the Student Dashboard to drive campus commerce.

---

AkadVerse AI Architecture -- v1.0