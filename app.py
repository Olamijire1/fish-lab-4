from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

# Define your route for the input page
@app.route('/')
def index():
    return render_template('input.html')

# Define your route for processing the input and showing the result
@app.route('/result', methods=['POST'])
def result():
    length1 = request.form['Length1']
    length2 = request.form['Length2']
    length3 = request.form['Length3']
    height = request.form['Height']
    width = request.form['Width']

    my_input = [length1, length2, length3, height, width]    # For example, let's just pass the input number to the result template
    my_input = np.array(my_input)
    my_input = my_input.reshape(1,-1)
    clf = load("model.joblib")
    result = clf.predict(pd.DataFrame(my_input))

    return render_template('output.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

