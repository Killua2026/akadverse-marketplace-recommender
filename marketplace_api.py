import os
import pandas as pd
from surprise import Reader, Dataset, SVD
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import uvicorn

# ------------------------------------------------------------------
# Environment & MongoDB connection
# ------------------------------------------------------------------
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable not set")

client = MongoClient(MONGO_URI)
db = client["akadverse_db"]
market_results_collection = db["marketplace_recommendations"]

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="AkadVerse Marketplace Recommender API")

# In-memory state
app.state.model = None
app.state.all_items = []
app.state.user_history = {}
app.state.item_sentiments = {}          # aggregate sentiment per item

# ------------------------------------------------------------------
# Event model
# ------------------------------------------------------------------
class PlatformEvent(BaseModel):
    event_type: str
    student_id: str
    payload: dict

# ------------------------------------------------------------------
# Global mock data (can be updated dynamically)
# ------------------------------------------------------------------
MOCK_DATA = {
    "student_id": [
        "23CE034397", "23CE034397", "23CE034397",
        "19MC022145", "19MC022145",
        "21EE044999", "21EE044999"
    ],
    "item_id": [
        "Acoustic Guitar Capo", "Advanced ML Course", "Horror Novel: The Shadows",
        "Advanced ML Course", "Videography Contract Service",
        "Acoustic Guitar Capo", "Horror Novel: The Shadows"
    ],
    "rating": [
        5.0, 4.5, 5.0,
        4.0, 5.0,
        4.5, 4.0
    ]
}

def load_and_train_model():
    """Trains the SVD model on the current MOCK_DATA with error resilience."""
    print("Refreshing AkadVerse marketplace AI model...")
    try:
        df = pd.DataFrame(MOCK_DATA)
        app.state.all_items = df['item_id'].unique().tolist()
        app.state.user_history = df.groupby('student_id')['item_id'].apply(list).to_dict()

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['student_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()

        algo = SVD()
        algo.fit(trainset)
        print("Model training complete.")
        return algo
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        return None

# Train initially on startup
app.state.model = load_and_train_model()

# ------------------------------------------------------------------
# Background tasks
# ------------------------------------------------------------------
def process_user_registration(student_id: str):
    """Initialize a new student profile (no retraining needed)."""
    print(f"[BACKGROUND] Initializing profile for new student {student_id}")
    if student_id not in app.state.user_history:
        app.state.user_history[student_id] = []
    # No model update – cold start handled by default scores

def process_click_event(student_id: str, item_id: str):
    """
    Track browsing behaviour – does NOT retrain the model.
    (In a real system you might store clicks in a separate collection for session-based recommendations.)
    """
    print(f"[BACKGROUND] Recording view for {student_id} on '{item_id}' (no model update)")

def process_order_completed(student_id: str, item_id: str, rating: float, sentiment_score: float = 0.0):
    """Append a new order and update aggregate sentiment – retrains the model."""
    print(f"[BACKGROUND] Processing order for {student_id} on '{item_id}' with sentiment {sentiment_score}")
    MOCK_DATA["student_id"].append(student_id)
    MOCK_DATA["item_id"].append(item_id)
    MOCK_DATA["rating"].append(rating)

    # Update sentiment (running average)
    current = app.state.item_sentiments.get(item_id, 0.0)
    app.state.item_sentiments[item_id] = (current + sentiment_score) / 2

    app.state.model = load_and_train_model()

def process_business_registered(student_id: str, business_name: str):
    """Add a new business (as an item) with a default rating of 5.0 from its owner."""
    print(f"[BACKGROUND] Adding business '{business_name}' for {student_id}")
    # Reuse order completion logic – this will retrain the model
    process_order_completed(student_id, business_name, 5.0, 0.0)

# ------------------------------------------------------------------
# Webhook endpoint (handles all AkadVerse event types)
# ------------------------------------------------------------------
@app.post("/webhook/event")
def handle_marketplace_event(event: PlatformEvent, background_tasks: BackgroundTasks):
    """Simulated Kafka consumer – routes events to appropriate background tasks."""
    if event.event_type == "order.completed":
        item_id = event.payload.get("item_id")
        rating = event.payload.get("rating", 5.0)
        sentiment = event.payload.get("sentiment", 0.0)

        if not item_id:
            raise HTTPException(status_code=400, detail="Missing 'item_id' in payload")

        background_tasks.add_task(
            process_order_completed,
            event.student_id, item_id, float(rating), float(sentiment)
        )
        return {"status": "success", "message": "Order queued for processing."}

    elif event.event_type == "click.event":
        item_id = event.payload.get("item_id")
        if not item_id:
            raise HTTPException(status_code=400, detail="Missing 'item_id'")
        background_tasks.add_task(process_click_event, event.student_id, item_id)
        return {"status": "success", "message": "Click recorded (no model update)."}

    elif event.event_type == "user.registered":
        background_tasks.add_task(process_user_registration, event.student_id)
        return {"status": "success", "message": "User initialized (no model update)."}

    elif event.event_type == "business.registered":
        biz_name = event.payload.get("business_name")
        if not biz_name:
            raise HTTPException(status_code=400, detail="Missing 'business_name'")
        background_tasks.add_task(process_business_registered, event.student_id, biz_name)
        return {"status": "success", "message": "Business added – model will retrain."}

    # Ignore events meant for other microservices
    return {"status": "ignored", "message": f"Event '{event.event_type}' handled elsewhere."}

# ------------------------------------------------------------------
# Prediction endpoints
# ------------------------------------------------------------------
@app.get("/predict-interest")
def predict_item_interest(student_id: str, item_id: str):
    """Predict how a specific student would rate a specific item (no sentiment)."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Recommendation engine is offline.")

    try:
        prediction = app.state.model.predict(uid=student_id, iid=item_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Student or item not found in training data")

    estimated_rating = round(prediction.est, 2)
    return {
        "student_id": student_id,
        "item_id": item_id,
        "predicted_rating": estimated_rating,
        "recommendation_strength": "High" if estimated_rating >= 4.0 else "Moderate/Low",
        "message": f"Based on campus trends, we predict {student_id} would rate this {estimated_rating}/5."
    }

@app.get("/top-recommendations")
def get_top_recommendations(student_id: str, top_k: int = 5):
    """
    Hybrid recommender: SVD predictions + Sentiment Boosting + Status Labeling.
    Results are saved to MongoDB for the Insight Engine and frontend.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Recommendation engine is offline.")

    bought_items = app.state.user_history.get(student_id, [])

    recommendations = []
    for item in app.state.all_items:
        pred = app.state.model.predict(uid=student_id, iid=item)
        base_score = pred.est

        # Apply sentiment-enriched ranking boost
        sentiment_bonus = app.state.item_sentiments.get(item, 0.0) * 0.2
        final_score = round(base_score + sentiment_bonus, 2)

        recommendations.append({
            "item_id": item,
            "score": final_score,
            "status": "Purchased" if item in bought_items else "Available"
        })

    # Sort by hybrid score
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    final_output = recommendations[:top_k]

    # Persistent sync to MongoDB Atlas
    sync_payload = {
        "student_id": student_id,
        "recommendations": final_output,
        "trigger": "api_request"
    }
    market_results_collection.update_one(
        {"student_id": student_id},
        {"$set": sync_payload},
        upsert=True
    )

    return {
        "student_id": student_id,
        "top_recommendations": final_output
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)