from flask import Flask ,request,render_template
import numpy as np
import pandas as pd
import pickle

#loading models
rfr = pickle.load(open('rfr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Predict',methods=['POST'])
def predict():
    if request.method=='POST':
        Crop_Year = request.form['Crop_Year']
        Area = request.form['Area']
        Production = request.form['Production']
        Annual_Rainfall = request.form['Annual_Rainfall']
        Fertilizer = request.form['Fertilizer']
        Pesticide = request.form['Pesticide']
        Crop = request.form['Crop']
        Season = request.form['Season']
        State = request.form['State']

        input_data = pd.DataFrame({
            'Crop_Year': [Crop_Year],
            'Area': [Area],
            'Production': [Production],
            'Annual_Rainfall': [Annual_Rainfall],
            'Fertilizer': [Fertilizer],
            'Pesticide': [Pesticide],
            'Crop': [Crop],
            'Season': [Season],
            'State': [State]
        })

        transformed_features = preprocessor.transform(input_data)
        predicted_value = rfr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html',predicted_value=predicted_value)

# python main
if __name__=='__main__':
    app.run(debug=True)