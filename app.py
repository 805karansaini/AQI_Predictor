from flask import Flask, render_template, request
import  joblib
import numpy as np

app = Flask(__name__)

lr = joblib.load("Models/linear_regression_model.pkl")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def main():
    if request.method == 'POST':
        T, TM, Tm, SLP, H, VV, V, VM = float(request.form['T']), float(request.form['TM']), float(request.form['Tm']), float(request.form['SLP']), float(request.form['H']), float(request.form['VV']), float(request.form['V']), float(request.form['VM'])
        lr_pm = lr.predict([[T, TM, Tm, SLP, H, VV, V, VM]])

        # print(lr_pm)

    return render_template("index.html", lr_pm = np.round(lr_pm,3))

if __name__ == "__main__":
    app.run(debug = True)
