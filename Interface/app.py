import streamlit as st
import requests
import pandas as pd
import atexit

# Register cleanup function with atexit
def on_exit():
    print("Shutting down Streamlit app...")
    try:
        requests.get("http://127.0.0.1:8000/terminate")  # Notify backend
    except requests.exceptions.RequestException:
        print("Failed to notify backend during shutdown.")

atexit.register(on_exit)

# Function to fetch dropdown options from the backend
@st.cache_data
def fetch_dropdown_options():
    try:
        response = requests.get("http://127.0.0.1:8000/get_dropdown_options")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch dropdown options: {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching dropdown options: {e}")
        return {}

# Fetch dropdown options
dropdown_options = fetch_dropdown_options()
stockcodes = dropdown_options.get("StockCode", [])
descriptions = dropdown_options.get("Description", [])
customer_ids = dropdown_options.get("CustomerID", [])
countries = dropdown_options.get("Country", [])

# Streamlit interface
st.title("Customer Clustering Prediction")
st.write("Enter the customer transaction details below and predict their cluster.")

# Input form
with st.form("input_form"):
    st.header("Enter Customer Transaction Details")
    Invoice = st.text_input("Invoice", placeholder="Enter Invoice ID (e.g., 536365)")

    # StockCode
    selected_stockcode = st.selectbox("Select StockCode", options=["Select"] + stockcodes)
    custom_stockcode = st.text_input("Or Enter StockCode", placeholder="Enter custom StockCode")
    StockCode = selected_stockcode if selected_stockcode != "Select" else custom_stockcode

    # Description
    selected_description = st.selectbox("Select Description", options=["Select"] + descriptions)
    custom_description = st.text_input("Or Enter Description", placeholder="Enter custom Description")
    Description = selected_description if selected_description != "Select" else custom_description

    Quantity = st.number_input("Quantity", min_value=1, step=1)
    InvoiceDate = st.text_input("Invoice Date (YYYY-MM-DD HH:MM)", placeholder="2011-12-09 14:30")
    Price = st.number_input("Price", min_value=0.01, step=0.01)

    # CustomerID
    selected_customer_id = st.selectbox("Select Customer ID", options=["Select"] + [str(cid) for cid in customer_ids])
    custom_customer_id = st.text_input("Or Enter Customer ID", placeholder="Enter custom Customer ID")
    CustomerID = str(selected_customer_id) if selected_customer_id != "Select" else str(custom_customer_id)

    # Country
    selected_country = st.selectbox("Select Country", options=["Select"] + countries)
    custom_country = st.text_input("Or Enter Country", placeholder="Enter custom Country")
    Country = selected_country if selected_country != "Select" else custom_country

    submitted = st.form_submit_button("Predict Cluster")

# Cluster descriptions
CLUSTER_DESCRIPTIONS = {
    0: "These are low-frequency, moderate-spending customers who purchase infrequently and have significant gaps between their shopping instances.",
    1: "These are inactive customers who made purchases in the past but have not returned for a significant period. They exhibit low frequency and moderate spending but had shorter gaps during active periods.",
    2: "These are top-tier, loyal, high-value customers. They are highly active, purchasing frequently with substantial spending, and they should be prioritized for retention efforts.",
    3: "These are medium-frequency, medium-spending customers. They are active but not highly frequent buyers, representing a key segment for nurturing and potential upselling.",
    4: "These are high-frequency, high-value customers. They shop often and spend significantly, making them another priority segment for loyalty programs and personalized marketing.",
}

# Predict and display cluster with description
if submitted:
    if not Invoice.strip() or not StockCode.strip() or not Description.strip() or not Country.strip():
        st.error("Please fill out all required fields.")
    else:
        try:
            # Ensure valid InvoiceDate format
            pd.to_datetime(InvoiceDate, format='%Y-%m-%d %H:%M')
            payload = {
                "Invoice": Invoice.strip(),
                "StockCode": StockCode.strip(),
                "Description": Description.strip(),
                "Quantity": Quantity,
                "InvoiceDate": InvoiceDate,
                "Price": Price,
                "CustomerID": CustomerID,
                "Country": Country.strip(),
            }

            with st.spinner("Predicting cluster..."):
                response = requests.post("http://127.0.0.1:8000/predict_cluster", json=payload)
                if response.status_code == 200:
                    cluster = response.json().get("Cluster")
                    if cluster is not None:
                        st.success(f"The predicted cluster is: {cluster}")
                        st.write(f"**Description**: {CLUSTER_DESCRIPTIONS.get(cluster, 'No description available.')}")
                    else:
                        st.error("Prediction failed. No cluster information returned by the server.")
                else:
                    st.error(f"Server error: {response.status_code} - {response.text}")
        except ValueError:
            st.error("Invalid InvoiceDate format. Please use 'YYYY-MM-DD HH:MM'.")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend server. Please ensure the API is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"An unexpected error occurred: {e}")
