import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model function with error handling
def load_model():
    try:
        with open('lenear_regression_model.pkl', 'rb') as file:
            model, scaler, le = pickle.load(file)
        return model, scaler, le
    except Exception as e:
        st.error("Error loading the model. Make sure the file exists and is correctly formatted.")
        return None, None, None

# Preprocessing function
def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform(data[['Extracurricular Activities']])  # Fix transformation
    df_transformed = scaler.transform(data)
    return df_transformed

# Prediction function
def predict_data(data):
    model, scaler, le = load_model()
    if model is None:
        return None

    preprocessed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(preprocessed_data)
    return prediction[0]

# Streamlit UI
def main():
    st.title('Student Performance Prediction')
    st.write('Enter your details for prediction:')

    # Input fields
    hours_studied = st.number_input("Hours studied", min_value=1, max_value=10, value=5)
    previous_scores = st.number_input("Previous Scores", min_value=40, max_value=100, value=70)
    extracurricular_activities = st.selectbox("Extracurricular Activities", ['YES', 'NO'])
    sleep_hours = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
    question_papers = st.number_input("Question Papers", min_value=0, max_value=10, value=5)

    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extracurricular_activities,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": question_papers
        }

        df = pd.DataFrame([user_data])  # Fix DataFrame creation

        prediction = predict_data(df)
        if prediction is not None:
            st.success(f"Your Predicted Score is: {prediction:.2f}")

if __name__ == '__main__':
    main()
