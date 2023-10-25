import pickle
from flask import Flask,request,app,jsonify,Response,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('ensemble_model.pkl','rb'))


@app.route('/predict_api',methods = ['POST'])
def predict_api():
    '''
    for direct  API call through request
    '''
    data = request.json(['data'])
    print(data)# return data will be key-pair value
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]# [0] will return predicted value
    # to convert into json
    return jsonify(output)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model.predict(final_features)[0]
    print('prediction class is:  ',output)
    
    result = ""
    if output == 1:
        result = "The credit card holder will be Defaulter in the next month"
    else:
        result = "The Credit card holder will not be Defaulter in the next month"
   
    return render_template('home.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
