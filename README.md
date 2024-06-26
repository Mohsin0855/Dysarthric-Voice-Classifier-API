# Dysarthric Voice Classifier API
This API provides a service for classifying audio files as either dysarthric or normal voice using machine learning models.
## Requirements
- Python 3.6+
- Flask
- Torch
- NumPy
- Joblib
- TorchVGGish
## Endpoint
- **POST /predict**

- **Request Body:** multipart/form-data containing a WAV audio file.

- **Response:** JSON object with a prediction label (Dysarthric or Normal voice) or an error message if the file format is invalid or if there is an error during prediction.
