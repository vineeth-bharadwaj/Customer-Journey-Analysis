from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import datetime as dt
import firebase_admin
from firebase_admin import credentials, firestore
from joblib import load
import os
import time

# Load the saved models and scaler
kmeans = load("kmeans_model.joblib") # Load the model that provided best results
# Make use of the same prerpocessing methods implemented during training
scaler = load("scaler.joblib") 
pca = load("pca.joblib")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("path/to/your_cloud_firestore_key_file.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the FastAPI app
app = FastAPI()

# Define the request schema
class InputData(BaseModel):
    Invoice: str
    StockCode: str
    Description: str
    Quantity: int
    InvoiceDate: str
    Price: float
    CustomerID: float
    Country: str


# Function to fetch existing customer data from Firestore
def fetch_existing_customer_data(customer_id):
    doc_ref = db.collection("onlineRetailTwo").document(str(customer_id))
    doc = doc_ref.get()
    if doc.exists:
        customer_data = doc.to_dict()
        invoices = customer_data.get("invoices", [])
        rows = []
        for invoice in invoices:
            for item in invoice.get("items", []):
                rows.append({
                    "Invoice": invoice["invoiceId"],
                    "StockCode": item["StockCode"],
                    "Description": item["Description"],
                    "Quantity": item["Quantity"],
                    "Price": item["Price"],
                    "CustomerID": customer_id,
                    "Country": customer_data.get("country"),
                    # Assume InvoiceDate is not available in the Firestore data
                    # Use placeholder or skip if unavailable
                    "InvoiceDate": dt.datetime.now().strftime('%Y-%m-%d %H:%M')
                })
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data exists


def update_firestore(customer_id, new_entry):
    try:
        doc_ref = db.collection("onlineRetailTwo").document(str(customer_id))
        doc = doc_ref.get()
        if doc.exists:
            # Existing customer data
            customer_data = doc.to_dict()
            invoices = customer_data.get("invoices", [])

            # Check if the invoice already exists
            existing_invoice = next((invoice for invoice in invoices if invoice["invoiceId"] == new_entry["Invoice"]), None)
            if existing_invoice:
                # Add the new item to the existing invoice
                existing_invoice["items"].append({
                    "StockCode": new_entry["StockCode"],
                    "Description": new_entry["Description"],
                    "Quantity": new_entry["Quantity"],
                    "Price": new_entry["Price"]
                })
            else:
                # Add a new invoice
                invoices.append({
                    "invoiceId": new_entry["Invoice"],
                    "items": [{
                        "StockCode": new_entry["StockCode"],
                        "Description": new_entry["Description"],
                        "Quantity": new_entry["Quantity"],
                        "Price": new_entry["Price"]
                    }]
                })

            # Update Firestore with the new invoices list
            doc_ref.update({"invoices": invoices})
        else:
            # If the customer doesn't exist, create a new document
            new_customer_data = {
                "customerId": customer_id,
                "country": new_entry["Country"],
                "invoices": [{
                    "invoiceId": new_entry["Invoice"],
                    "items": [{
                        "StockCode": new_entry["StockCode"],
                        "Description": new_entry["Description"],
                        "Quantity": new_entry["Quantity"],
                        "Price": new_entry["Price"]
                    }]
                }]
            }
            doc_ref.set(new_customer_data)
        print(f"Successfully updated Firestore for Customer ID: {customer_id}")
    except Exception as e:
        print(f"Failed to update Firestore for Customer ID: {customer_id}. Error: {e}")



# Preprocess data function
def preprocess_data(new_data):
    # Convert new data to DataFrame
    new_df = pd.DataFrame([new_data.dict()])

    # Fetch existing customer data from Firestore
    existing_data = fetch_existing_customer_data(new_data.CustomerID)

    # Combine existing and new data
    combined_data = pd.concat([existing_data, new_df], ignore_index=True)

    # Create Revenue
    combined_data["Revenue"] = combined_data["Quantity"] * combined_data["Price"]

    # Convert InvoiceDate to datetime
    combined_data['InvoiceDate'] = pd.to_datetime(combined_data['InvoiceDate'], format='%Y-%m-%d %H:%M', errors='coerce')
    latest_date = dt.datetime(2011, 12, 10, 0, 0)  # Include time (00:00)

    # Group by Customer ID to calculate RFM metrics
    RFM = combined_data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'Invoice': 'nunique',
        'Revenue': 'sum'
    }).reset_index()
    RFM.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Revenue': 'Monetary'}, inplace=True)

    # Add Shopping_Cycle and Interpurchase_Time
    RFM["Shopping_Cycle"] = RFM["Recency"]  # Fallback to Recency as a placeholder
    RFM["Interpurchase_Time"] = RFM["Shopping_Cycle"] // RFM["Frequency"]

    # Scale the features
    RFMT = RFM[["Recency", "Frequency", "Monetary", "Interpurchase_Time"]]
    RFMT_scaled = scaler.transform(RFMT)

    # Apply PCA
    RFMT_pca = pca.transform(RFMT_scaled)

    return RFMT_pca

@app.get("/get_dropdown_options")
def get_dropdown_options():
    try:
        stockcodes = set()
        descriptions = set()
        customer_ids = set()
        countries = set()

        # Fetch all documents from the Firestore collection
        docs = db.collection("onlineRetailTwo").stream()
        for doc in docs:
            doc_id = doc.id  # Extract the document ID as the customer ID
            data = doc.to_dict()

            # Add the document ID (customer ID) to the set
            customer_ids.add(doc_id)

            # Extract other fields from the document
            countries.add(data.get("country", "Unknown"))
            for invoice in data.get("invoices", []):
                for item in invoice.get("items", []):
                    stockcodes.add(item.get("StockCode"))
                    descriptions.add(item.get("Description"))

        # Return dropdown options
        return {
            "StockCode": list(stockcodes),
            "Description": list(descriptions),
            "CustomerID": list(customer_ids),
            "Country": list(countries),
        }
    except Exception as e:
        return {"error": str(e)}


# Define the prediction endpoint
@app.post("/predict_cluster")
def predict_cluster(data: InputData):
    # Preprocess the input data
    RFMT_pca = preprocess_data(data)

    # Predict cluster
    cluster = kmeans.predict(RFMT_pca)

    # Update Firestore with the new entry
    update_firestore(data.CustomerID, data.dict())

    return {"Cluster": int(cluster[0])}


@app.on_event("shutdown")
async def shutdown_event():
    # Perform cleanup tasks if needed (e.g., closing database connections)
    print("Shutting down FastAPI server...")


@app.get("/terminate")
async def terminate(background_tasks: BackgroundTasks):
    # Use a background task to delay shutdown to allow response to complete
    background_tasks.add_task(shutdown_server)
    return JSONResponse(content={"message": "Shutting down server..."})

def shutdown_server():
    time.sleep(1)  # Allow response to be sent before termination
    os._exit(0)  # Forcefully terminate the server process
