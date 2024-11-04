from flask import Flask, render_template, request
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('loan_prediction_model.pkl')
except FileNotFoundError:
    model = None
    print("Error: 'loan_prediction_model.pkl' file not found.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    scaler = None
    print("Error: 'scaler.pkl' file not found.")
except Exception as e:
    scaler = None
    print(f"Error loading scaler: {e}")

@app.route('/')
def home():
    # Pass an empty dictionary for data to avoid 'data' being undefined
    return render_template('index.html', data={})

@app.route('/predict', methods=['POST'])
def predict():
    # Return error if model or scaler is not loaded
    if not model:
        return render_template('index.html', prediction_text="Model file is missing or failed to load.", data={})
    if not scaler:
        return render_template('index.html', prediction_text="Scaler file is missing or failed to load.", data={})
    
    # Collect form data
    data = {field: request.form.get(field, '') for field in request.form.keys()}
    
    try:
        # Prepare data for model
        input_data = {
            'no_of_dependents': int(data['no_of_dependents']),
            'education': 1 if data['education'].lower() == 'graduate' else 0,
            'self_employed': 1 if data['self_employed'].lower() == 'yes' else 0,
            'income_annum': float(data['income_annum']),
            'loan_amount': float(data['loan_amount']),
            'loan_term': float(data['loan_term']),
            'cibil_score': int(data['cibil_score']),
            'residential_assets_value': float(data['residential_assets_value']),
            'commercial_assets_value': float(data['commercial_assets_value']),
            'luxury_assets_value': float(data['luxury_assets_value']),
            'bank_asset_value': float(data['bank_asset_value'])
        }

        # Feature engineering
        input_data['income_loan_ratio'] = input_data['income_annum'] / input_data['loan_amount']
        input_data['cibil_loan_term'] = input_data['cibil_score'] * input_data['loan_term']

        # Convert to DataFrame and drop engineered columns before scaling
        input_df = pd.DataFrame([input_data])
        input_df = input_df.drop(columns=['income_loan_ratio', 'cibil_loan_term'])

        # Scale the input data and make prediction
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = 'Rejected' if prediction == 1 else 'Approved'

        return render_template('index.html', prediction_text=f'Loan Prediction: {result}', data=data)

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return render_template('index.html', prediction_text=f"An error occurred: {e}", data=data)

if __name__ == '__main__':
    app.run(debug=True)
