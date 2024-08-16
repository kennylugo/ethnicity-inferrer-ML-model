from flask import Flask, request, jsonify
import joblib
import pandas as pd
from werkzeug.utils import secure_filename
import os
from classification import classify_dna_sample
from preprocessing import run_preprocessing_logic

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the necessary files are included in your app's directory
processed_data_path = 'processed_data.pkl'
model_path = 'DecisionTree_classifier_model.bin'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load pre-trained model
trained_model = joblib.load(model_path)

# Preprocess the data
if os.path.exists(processed_data_path):
    processed_genotype_data, feature_matrix, aims_data_df = joblib.load(processed_data_path)

snp_list = aims_data_df['Position'].tolist()
ethnicity_labels = trained_model.classes_

@app.route('/classify', methods=['POST'])
def classify():
    print("running /classify")
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:

            print("Uploading file")
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            print("About to perform classification")

            # Perform classification
            top_3_ethnicities = classify_dna_sample(file_path, trained_model, snp_list, ethnicity_labels)
            print(f"Top 3 Predicted Ethnicities: {top_3_ethnicities}")

            # Delete the DNA file after classification
            os.remove(file_path)

            return jsonify({"top_3_ethnicities": top_3_ethnicities})
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/', methods=['GET'])
def index():
    return "Welcome to the DNA Classifier API!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7300))
    app.run(host='0.0.0.0', port=port, debug=False)
