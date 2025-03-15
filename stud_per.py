import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://tauseefarif29:tauseefarif29@cluster0.3hehk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client['student']
collection = db["student_pred"]


def load_model():
    with open('linear_regression_model.pkl','rb') as file :
        model , scaler ,le =  pickle.load(file)
    return model , scaler ,le

def preprocessing_input_data(data,scaler ,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed =scaler.transform(df)
    return df_transformed

def predict_data(data):
    model , scaler ,le =load_model()
    preprocessed_data = preprocessing_input_data(data,scaler ,le)
    prediction = model.predict(preprocessed_data)
    return round(prediction[0],2)

def main():
    st.title("Student preformance prediction")
    st.write("enter your detail for get prediction on your input")
    
    hour_studied = st.number_input("Hours studied",min_value=1,max_value= 10 ,value = 5)
    Previous_Scores= st.number_input("Previous Scores",min_value = 40 , max_value= 100 , value = 70)
    Extracurricular_Activities= st.selectbox("Extracurricular Activities",["Yes" ,"No"])
    Sleep_Hours = st.number_input("Sleep Hours" , min_value= 4 , max_value= 10 , value = 7)
    Question_Papers  = st.number_input("Question Papers" , min_value = 0 , max_value= 10 , value = 5 )
    
    
    if st.button("predict_your_score"):
        user_data = {
            "Hours Studied":hour_studied,
            "Previous Scores":Previous_Scores,
            "Extracurricular Activities":Extracurricular_Activities,
            "Sleep Hours":Sleep_Hours,
            "Sample Question Papers Practiced":Question_Papers
            
        }
        prediction = predict_data(user_data)
        st.success(f"your prediction result is:{prediction} ")
        user_data["prediction"] = float(prediction)

        # ✅ Convert all NumPy types to native Python types
        user_data = {key: int(value) if isinstance(value, (np.int64, np.int32)) else float(value) if isinstance(value, (np.float64, np.float32)) else value for key, value in user_data.items()} 

        collection.insert_one(user_data)  # ✅ Fixed TypeError

        
       
    
if __name__ == "__main__":
    main()
    

