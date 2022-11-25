from flask import Flask,render_template,request,redirect
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

@app.route('/',methods=["GET"])
def homepage():
    return render_template('homepage.html')

@app.route('/teampage',methods=["GET"])
def teampage():
    return render_template('teampage.html')

@app.route('/prediction',methods=["POST","GET"])
def prediction():
    if request.method == "POST":
        string = request.form['val']
        if(string ==""):
            return render_template('prediction.html')            
        string = string.split(',')
        x_input = [eval(i) for i in string]
        sc = StandardScaler()
        x_input = sc.fit_transform(np.array(x_input).reshape(-1,1))
        x_input = np.array(x_input).reshape(1,-1)
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,10,1))
        model = load_model('crude_oil.h5')
        output = model.predict(x_input)
        val = sc.inverse_transform(output)
        return render_template('prediction.html' , prediction = round(val[0][0],2))  
    if request.method == "GET":    
        return render_template('prediction.html')    
    if request.method == "GET":    
        return render_template('teampage.html')

if __name__=="__main__":
    model = load_model('crude_oil.h5')
    app.run(debug=True)