import os
import torch
import numpy as np
from torchvggish import vggish, vggish_input
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize the VGGish model
model = vggish()
model.eval()

# Load the RandomForestClassifier model
loaded_model = joblib.load("random_forest_model.pkl")

# Function to extract features from audio using VGGish
def extract_embedding_from_audio(audio_path, expected_features=384):
    try:
        # Normalize the file path
        audio_path = os.path.normpath(audio_path)
        
        # Perform feature extraction
        example = vggish_input.wavfile_to_examples(audio_path)
        
        # Check if feature extraction was successful
        if example.nelement() == 0:
            return None
        
        # Perform model inference
        embeddings = model.forward(example)
        
        # Convert embeddings to NumPy array
        embeddings_np = embeddings.detach().numpy()
        
        # Perform mean pooling to aggregate features across time
        aggregated_features = np.mean(embeddings_np, axis=0)
        
        # Resize features to match the expected number of features
        if aggregated_features.shape[0] >= expected_features:
            # If there are enough features, truncate to the expected number
            extracted_features = aggregated_features[:expected_features]
        else:
            # If there are fewer features, pad with zeros
            padding = expected_features - aggregated_features.shape[0]
            extracted_features = np.pad(aggregated_features, ((0, padding)), mode='constant', constant_values=0)
        
        return extracted_features.tolist()
    except Exception as e:
        print("Error:", e)
        return None

# Initialize the Flask application
app = Flask(__name__)

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract features from the audio file
        features = extract_embedding_from_audio(filepath)

        if features is not None:
            # Make predictions using the loaded RandomForestClassifier model
            predictions = loaded_model.predict([features])

            # Map the predicted label to dysarthric or normal voice
            predicted_label = "Dysarthric" if predictions[0] == 1 else "Normal voice"

            # Return the prediction
            return jsonify({'prediction': predicted_label})
        else:
            return jsonify({'error': 'Error extracting features from audio'})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
