from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os
import warnings

# Suppress the feature names warning
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
CORS(app)

def load_model():
    """Load the trained model with proper error handling"""
    try:
        model_path = "trained_model.pkl"
        
        if not os.path.exists(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "trained_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        print(f"📊 Model expects {model.n_features_in_} features")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_model_reg():
    """Load the logistic regression model with proper error handling"""
    try:
        model_path = "trained_model_log.pkl"
        
        if not os.path.exists(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "trained_model_log.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Logistic regression model file not found at: {model_path}")
        
        model = joblib.load(model_path)
        print("✅ Logistic regression model loaded successfully!")
        print(f"📊 Logistic model expects {model.n_features_in_} features")
        return model
        
    except Exception as e:
        print(f"❌ Error loading logistic regression model: {e}")
        return None

# Load the model
model = load_model()
logreg = load_model_reg()
if model is None or logreg is None:
    print("Server cannot start without the model files.")
    exit(1)

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Check if data is received
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        # Check for required fields
        required_fields = ['study_hours', 'attendance', 'practice_tests']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Get and validate values
        try:
            study_hours = float(data['study_hours'])
            attendance = float(data['attendance'])
            practice_tests = int(data['practice_tests'])
        except (ValueError, TypeError):
            return jsonify({'error': 'All fields must be valid numbers'}), 400
        
        # Validate ranges
        if study_hours < 0 or study_hours > 24:
            return jsonify({'error': 'Study hours must be between 0 and 24'}), 400
        if attendance < 0 or attendance > 100:
            return jsonify({'error': 'Attendance must be between 0 and 100'}), 400
        if practice_tests < 0 or practice_tests > 10:
            return jsonify({'error': 'Practice tests must be between 0 and 10'}), 400
        
        # Prepare input data - FIXED: Ensure correct shape and type
        input_data = np.array([[study_hours, attendance, practice_tests]], dtype=float)
        
        # Make prediction with Linear Regression
        score_prediction = float(model.predict(input_data)[0])
        score_prediction = max(0, min(100, round(score_prediction, 2)))
        
        # Make prediction with Logistic Regression - FIXED: Use same input format
        pass_prediction = int(logreg.predict(input_data)[0])
        
        # Prepare response
        if pass_prediction == 1:
            passed_status = "Status: Passed 🎉"
            message_text = "Congratulations! You passed the exam 🎉"
        else:
            passed_status = "Status: Failed 😔"
            message_text = "Unfortunately, you did not pass. Keep trying! 💪"
        
        return jsonify({
            'score': score_prediction,
            'passed': passed_status,
            'message': message_text
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")  # This will show us the actual error
        return jsonify({'error': 'Prediction failed. Please check server logs.'}), 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    print("🚀 Starting Student Performance Predictor Server...")
    print("🌐 Server running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)