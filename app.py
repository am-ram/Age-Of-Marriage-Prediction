from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    print("Request Received")
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    predicted_age_of_marriage = model.predict([[int(request.args['gender']),
                                                int(request.args['religion']),
                                                int(request.args['caste']),
                                                int(request.args['mother_tongue']),
                                                int(request.args['country']), ]])
    #return str(round(predicted_age_of_marriage[0], 2))
    val=str(round(predicted_age_of_marriage[0], 2))
    return render_template('index.html',predicted_age_of_marriage=val)

if __name__ == '__main__':
    app.run(debug=True)
