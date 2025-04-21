import os

os.system("pip install scikit-learn")

import gradio as gr
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('diseases2.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset to fit the scaler
data = pd.read_csv("diabetes.csv")
X = data.drop(columns="Outcome", axis=1)

# Fit the StandardScaler on dataset features
scaler = StandardScaler()
scaler.fit(X)

# Prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Standardize input data
    std_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(std_data)

    # Return result
    if prediction[0] == 1:
        return " The person is Diabetic**."
    else:
        return " The person is Not Diabetic."

# Create Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=1, minimum=0, maximum=20),
        gr.Number(label="Glucose Level", value=100, minimum=0, maximum=300),
        gr.Number(label="Blood Pressure", value=80, minimum=0, maximum=200),
        gr.Number(label="Skin Thickness", value=20, minimum=0, maximum=100),
        gr.Number(label="Insulin Level", value=80, minimum=0, maximum=900),
        gr.Number(label="BMI", value=25.0, minimum=0.0, maximum=70.0),
        gr.Number(label="Diabetes Pedigree Function", value=0.5, minimum=0.0, maximum=2.5),
        gr.Number(label="Age", value=30, minimum=1, maximum=120),
    ],
    outputs="text",
    title="Diabetes Prediction ",
    description="Enter the required details to predict whether a person has diabetes."
)

# Launch Gradio app
if __name__ == "__main__":
    interface.launch()
