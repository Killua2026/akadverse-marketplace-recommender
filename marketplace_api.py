import pandas as pd # For data manipulation and creating the mock dataset
from surprise import Reader, Dataset, SVD # For building the collaborative filtering model
from fastapi import FastAPI # For creating the API endpoints
import uvicorn # For running the FastAPI server


# Initialize the FastAPI application for the Marketplace
app = FastAPI(title="AkadVerse Marketplace Recommender API")

# We will store our trained model and campus catalog in memory
app.state.model = None
app.state.all_items = []
app.state.user_history = {}

def load_and_train_model():
    """
    Simulates fetching order history from PostgreSQL and trains the 
    Collaborative Filtering model using the Surprise library. [cite: 162, 163]
    """
    print("Loading campus marketplace data...")
    
    # 1. Create a mock dataset of student purchases/ratings (1 to 5 scale)
    mock_data = {
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
    
    df = pd.DataFrame(mock_data)
    
    # --- NEW CATALOG LOGIC ---
    # Extract all unique items in the marketplace
    app.state.all_items = df['item_id'].unique().tolist()
    
    # Group by student to see what they already bought so we don't recommend it again
    # This creates a dictionary: {'23CE034397': ['Acoustic Guitar Capo', ...], ...}
    app.state.user_history = df.groupby('student_id')['item_id'].apply(list).to_dict()
    # -------------------------

    # 2. Prepare the data for the Surprise library
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['student_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # 3. Initialize and train the SVD (Singular Value Decomposition) algorithm
    print("Training the SVD Collaborative Filtering model...")
    algo = SVD()
    algo.fit(trainset)
    
    print("Model training complete!")
    return algo

# Train the model immediately when the server starts
app.state.model = load_and_train_model()

@app.get("/predict-interest")
def predict_item_interest(student_id: str, item_id: str):
    """
    Endpoint to predict how much a specific student will like a specific item.
    """
    if app.state.model is None:
        return {"error": "Model is not trained yet."}
        
    try:
        prediction = app.state.model.predict(uid=student_id, iid=item_id)
    except:
        return {"error": "Student ID or Item ID not found in training data"}
        
    estimated_rating = round(prediction.est, 2)
    
    return {
        "student_id": student_id,
        "item_id": item_id,
        "predicted_rating": estimated_rating,
        "recommendation_strength": "High" if estimated_rating >= 4.0 else "Moderate/Low",
        "message": f"Based on campus trends, we predict {student_id} would rate this {estimated_rating}/5."
    }

# --- NEW ENDPOINT ---
@app.get("/top-recommendations")
def get_top_recommendations(student_id: str, top_k: int = 3):
    """
    Scans the entire marketplace, filters out items the student already owns,
    predicts their interest in the remaining items, and returns the top matches.
    """
    if app.state.model is None:
        return {"error": "Model is not trained yet."}
        
    # 1. Get items the student has already interacted with (default to empty list if new)
    bought_items = app.state.user_history.get(student_id, [])
    
    # 2. Filter out items they already own
    unseen_items = [item for item in app.state.all_items if item not in bought_items]
    
    # 3. Predict ratings for all unseen items
    predictions = []
    for item in unseen_items:
        pred = app.state.model.predict(uid=student_id, iid=item)
        predictions.append({
            "item_id": item,
            "predicted_rating": round(pred.est, 2)
        })
        
    # 4. Sort the list by the highest predicted rating
    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    
    return {
        "student_id": student_id,
        "already_owned": bought_items,
        "top_recommendations": predictions[:top_k]
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8001)