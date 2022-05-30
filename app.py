import logging.config

import numpy as np
from flask import Flask
from flask import render_template, request, redirect, url_for
from src.train import get_model, predict_ind
from src.createdb import Transaction, ResponseManager

# Initialize Flask app
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG from flask_config.py
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])

# Load model and encoder to make new predictions
model_path = app.config["MODEL_PATH"]
encoder_path = app.config["ENCODER_PATH"]
model, enc = get_model(model_path, encoder_path)

# Manager to query data from sql table
response_manager = ResponseManager(app)

@app.route("/", methods=['GET', 'POST'])
def home():
    '''Main page of application providing information and collecting form info
    Args:
        None
    Returns:
        rendered html template
    '''
    if request.method == "GET":
        try:
            logger.info("Main page returned")
            return render_template('index.html')
        except Exception as error:
            logger.error("Error page returned with error: %s", error)
            return render_template('error.html')

    if request.method == "POST":
        try:
            gender = int(request.form["gender"])
            race = int(request.form["race"])
            educ = int(request.form["education"])
            marital = int(request.form["ms"])
            work = int(request.form["work"])
            region = int(request.form["region"])
            cat_vars = [gender, race, educ, marital, work, region]
            year = int(request.form["year"])
            prediction = predict_ind(model, enc, cat_vars, year)
            top3 = np.argsort(prediction)[::-1]  # Gets indices sorted array descending
            top3 = top3[:3]  # Gets top 3 highest probabilities indexes
            top3_probs = [np.round(prediction[i], 2) for i in top3]  # Gets top 3 highest probabilities
            url_for_post = url_for('response_page', class1=top3[0], class2=top3[1],
                                   class3=top3[2], prob1=top3_probs[0],
                                   prob2=top3_probs[1], prob3=top3_probs[2])
            logger.info("Prediction submitted from form")
            return redirect(url_for_post)
        except Exception as error:
            logger.error("Error page returned with error: %s", error)
            return render_template('error.html')

@app.route("/response.html/<class1>/<class2>/<class3>/<prob1>/<prob2>/<prob3>",
           methods=['GET', 'POST'])
def response_page(class1, class2, class3, prob1, prob2, prob3):
    '''Page that displays model predictions and sql table with additional info
    Args:
        class1 (str): number indicating predicted reason number one
        class2 (str): number indicating predicted reason number two
        class3 (str): number indicating predicted reason number three
        prob1 (str): probability for predicted reason number one
        prob2 (str): probability for predicted reason number two
        prob3 (str): probability for predicted reason number three
    Returns:
        rendered html template
    '''
    if request.method == "GET":
        try:
            response = response_manager.session.query(Transaction)\
                                       .filter(Transaction.output.in_([int(class1), int(class2), int(class3)]))
            probs = [prob1, prob2, prob3]
            logger.info("Response page requested")
            return render_template('response.html', responses=response,
                                   probabilities=probs)
        except Exception as error:
            logger.error("Error getting page: %s", error)
            logger.debug("Make sure to fill entire form")
            return render_template('error.html')

    if request.method == "POST":
        url_for_post = url_for('home/')
        return redirect(url_for_post)


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
