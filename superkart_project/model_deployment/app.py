import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
model = joblib.load('best_random_forest_model.joblib')

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class Item(BaseModel:
    Product_Weight: float
    Product_Sugar_Content: int
    Product_Allocated_Area: float
    Product_Type: int
    Product_MRP: float
    Store_Establishment_Year: int
    Store_Size: int
    Store_Location_City_Type: int
    Store_Type: int

@app.get("/")
async def read_root():
    return {"message": "Sales Forecasting Model API"}

@app.post("/predict/")
async def predict_sales(item: Item):
    # Convert input to DataFrame
    df = pd.DataFrame([item.dict()])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    return {"predicted_sales": prediction}
