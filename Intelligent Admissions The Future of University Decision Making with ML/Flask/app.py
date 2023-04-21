import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
from tensorflow.keras.models import load_model
model = pickle.load(open('university.pkl','rb'))
# model = load_model('model.h5')
@app.route('/')
def home():
  return render_template('Demo2.html')
# pickle.dump(lr, open('university.pkl','wb'))
@app.route('/y_predict',methods = ['post'])
def y_predict():
  #for rendering results on html gui
  #min max scaling
  min1 = [290.0, 92.0, 1.0, 1.0, 6.8, 0.0]
  max1 = [340.0, 120.0, 5.0, 5.0, 9.92, 1.0]
  k = [float(x) for x in request.form.values()]
  p = [1]
  for i in range(6):
    l = (k[i]-min1[i])/(max1[i]-min1[i])
    p.append(l)
  prediction = model.predict([p])
#   print(prediction)
  output = prediction[0]
  if(output == False):
    return render_template('noChance.html')
  else:
    return render_template('chance.html')
if __name__ == "__main__":
  app.run(debug=False)