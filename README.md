# Customer-Journey-Analysis
A project on customer segmentation using different clustering methods. 

# Install dependencies:
The interface folder has the requirements.txt

```pip install -r requirements.txt```

# Download the Instacart Dataset and Online Retail Dataset
Here are the links from which the data was used

Instacart: 
https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Online Retail:
https://www.kaggle.com/datasets/vijayuv/onlineretail

# Running the interface:
The Cloud Firestore key was generated and added in directory that contained main.py. The data being used was pushed to the cloud firestore (in my case the online retail data). The code I used is in the clustering folder.

Run the backend file, main.py, with the command:
```uvicorn main:app --reload```

Run the frontend file, app.py, with the command:
```streamlit run app.py```

While closing, close app.py and then close main.py at the terminal window through Ctrl + C. 
