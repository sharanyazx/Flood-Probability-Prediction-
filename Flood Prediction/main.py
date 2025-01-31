from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Define a function to preprocess the input data
def preprocess_input(data):
    # Drop unnecessary columns
    data.rename(columns={"daily_runoff": "daily runoff", "discharge": "Discharge"}, inplace=True)
    
    return data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit_data", methods=["POST"])

def submit_data():
    # Parse form data
    river = request.form["river"]
    print("Loading result for",river,"river")
    # Load the trained machine learning model
    model = joblib.load("Trained/" + river + "_trained_model.pkl")

    discharge = float(request.form["discharge"])
    daily_runoff = float(request.form["daily_runoff"])
    weekly_runoff = float(request.form["weekly_runoff"])

    # Preprocess data
    input_data = pd.DataFrame({

    "Discharge": [discharge],  # Adjusted feature name
    "daily runoff": [daily_runoff],  # Adjusted feature name
    "weekly runoff": [weekly_runoff],  # Adjusted feature name
})



    input_data=preprocess_input(pd.DataFrame(input_data))

    # Make predictions
    prediction = model.predict(input_data)

    # Return prediction results

    return f"Flood Prediction for {river}: {bool(prediction)}"

if __name__ == "__main__":
    app.run(debug=True,port =5500)
