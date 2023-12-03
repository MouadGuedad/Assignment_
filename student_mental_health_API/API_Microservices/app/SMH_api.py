import os
print(os.getcwd())
import uvicorn
from fastapi import FastAPI
from variables import StudentVariables  
import numpy
import pickle
import onnxruntime as rt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

# Create app object 
app = FastAPI()

# Load model scalar
pickle_in = open("preprocessing_transformer.pkl", "rb")
preprocessing_transformer = pickle.load(pickle_in)

# Load the model
sess = rt.InferenceSession("best_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# API Endpoints
@app.get('/')
def index():
    return {'Hello': 'Welcome to Student Mental health prediction service, access the API docs and test the API at http://0.0.0.0/docs.'}

@app.post('/predict')
def predict_SMH(data: StudentVariables):
    data = data.dict()

    # fetch input data using data variables
    gender = data['gender']
    age = data['age']
    course = data['course']
    year = data['year']
    cgpa = data['cgpa']
    marital_status = data['marital_status']
    depression = data['depression']
    anxiety = data['anxiety']
    panic_attack = data['panic_attack']
    treatment = data['treatment']

    data_to_pred = numpy.array([[gender, age, course, year,
                                 cgpa, marital_status, depression, anxiety, panic_attack, treatment]])

    # Scale input data
    data_to_pred = preprocessing_transformer.fit_transform(data_to_pred.reshape(1, 10))

    # Model inference
    prediction = sess.run(
        [label_name], {input_name: data_to_pred.astype(numpy.float32)})[0]

    
    return {
        'prediction': prediction
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
