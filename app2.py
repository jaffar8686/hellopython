from flask import Flask,request,render_template,redirect,jsonify
import numpy as np
import pickle

app=Flask(__name__,template_folder='template')






regress=pickle.load(open('model_iris.pkl','rb'))




#print(regress.predict([[5.1,3.5,1.4,0.2]]))

@app.route('/')
def home():
    return render_template('test.html')
@app.route('/predicted',methods=['POST'])
def predicted():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regress.predict(final_features)
    #print(prediction[0])

    output = round(prediction[0],2)

    return render_template('test.html', prediction_text='iris species is {}'.format(output),output=output)


if __name__=='__main__':
    app.run(debug=True)
  