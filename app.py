from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('climate_change_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debugging line

        # Extract and convert features from the JSON input
        mei = float(data.get('MEI'))
        co2 = float(data.get('CO2'))
        ch4 = float(data.get('CH4'))
        n2o = float(data.get('N2O'))
        cfc_11 = float(data.get('CFC-11'))
        cfc_12 = float(data.get('CFC-12'))
        tsi = float(data.get('TSI'))
        aerosols = float(data.get('Aerosols'))

        # Prepare input for the model
        features = np.array([[mei, co2, ch4, n2o, cfc_11, cfc_12, tsi, aerosols]])
        print("Features array:", features)  # Debugging line

        # Predict temperature
        predicted_temp = model.predict(features)
        print("Predicted temperature:", predicted_temp)  # Debugging line

        # Return the prediction as JSON
        return jsonify({'predicted_temperature': predicted_temp[0][0]})
    except Exception as e:
        print("Error:", e)  # Debugging line
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
