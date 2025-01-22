import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

# Initialize Firebase Admin SDK
cred = credentials.Certificate("path/to/your_cloud_firestore_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the local dataset
file_path = "onlineretail/onlineRetailDataForInterface.csv"
data = pd.read_csv(file_path, encoding="unicode_escape")

# Clean before pushing to Firebase to save your write quota
data.dropna(subset=["Customer ID"], axis=0, inplace=True)
data = data[~data.Invoice.str.contains('C', na=False)]
data = data.drop_duplicates(keep="first")
data = data[data.Quantity > 0]
data = data[data.Price > 0]

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return up_limit, low_limit

def replace_with_threshold(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# Remove outliers
replace_with_threshold(data, "Quantity")
replace_with_threshold(data, "Price")

# Ensure column names match Firestore naming conventions
data.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
collection_name = "onlineRetailTwo"
grouped_customers = data.groupby("Customer_ID")
write_count = 0
max_writes = 19000  # Write quota is 20K/day.

# Upload each customer to Firestore
for customer_id, customer_data in grouped_customers:
    if write_count >= max_writes:
        break
    first_row = customer_data.iloc[0]  # Get the first row for country
    country = first_row["Country"]
    grouped_invoices = customer_data.groupby("Invoice")
    invoices = []
    for invoice_id, invoice_data in grouped_invoices:
        items = invoice_data[["StockCode", "Description", "Quantity", "Price", "InvoiceDate"]].to_dict(orient="records")
        invoices.append({
            "invoiceId": str(invoice_id),
            "items": items
        })
    customer_doc = {
        "country": country,
        "invoices": invoices
    }

    try:
        # Save to Firestore
        db.collection(collection_name).document(str(customer_id)).set(customer_doc)
        write_count += 1
    except ResourceExhausted:
        print("Write quota exceeded. Terminating program.")
        break
    except GoogleAPICallError as e:
        print(f"An error occurred: {e}. Terminating program.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Continuing with the next record.")

print("Data upload completed or terminated due to quota limits.")
