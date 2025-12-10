<<<<<<< HEAD
# Nail Disease Classification

A comprehensive deep learning system for detecting and classifying nail diseases using multiple state-of-the-art computer vision models with an interactive web interface.
=======
# Nail Disease Screening Application

A full end-to-end machine learning web application for nail disease screening (not a medical diagnosis tool).
>>>>>>> feature/flask-web-application

## ⚠️ Medical Disclaimer

**This application is for informational purposes only and is not a substitute for professional medical diagnosis, treatment, or advice.** Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

<<<<<<< HEAD
## Overview

This project implements a multi-model approach to nail disease classification, combining object detection and classification techniques to provide accurate diagnosis of various nail conditions. The system includes five different classification models, YOLO-based nail detection, GradCAM visualization for model interpretability, and a user-friendly web interface.

## Features

- **Multiple Classification Models**: Five different CNN architectures for robust disease classification
  - AlexNet
  - VGG16
  - ResNet50
  - DenseNet
  - GoogleNet
- **Object Detection**: YOLO-based nail region detection
- **Model Interpretability**: GradCAM visualization to understand model predictions
- **Web Interface**: Interactive Flask-based UI for easy image upload and analysis
- **Real-time Predictions**: Quick disease classification with confidence scores

## Dataset
The models are trained on nail disease datasets containing images of various nail conditions. The datasets can be accessed from multiple sources:
Classification Dataset

Kaggle Nail Disease Dataset: https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset/data 

Contains labeled images of different nail diseases
Used for training AlexNet, VGG16, ResNet50, DenseNet, and GoogleNet models



## Object Detection Dataset

Roboflow Fingernails Dataset: https://universe.roboflow.com/fingernail-ztwys/fingernails-xb812

Contains annotated nail images for object detection
Used for training the YOLO nail detection model



Setup Instructions
For Classification Models (Kaggle):

Download the dataset from Kaggle
Extract the files to a data/ directory in the project root
Ensure the dataset is organized with separate folders for each disease class
Update the dataset paths in the respective notebook files in the `notebooks/` directory before training
Note: You may need to create a Kaggle account and accept the dataset's terms of use

For YOLO Detection (Roboflow):

Access the Roboflow dataset at the link above
Generate and download the dataset in YOLO format
Place the dataset in the appropriate directory structure
Update the data configuration in `notebooks/yolo_nail_object_detection.ipynb`
The dataset includes train, validation, and test splits

## Project Structure

```
NailDiseaseClassification/
├── notebooks/                       # Jupyter notebooks for model training
│   ├── alexnet-nailclassification.ipynb
│   ├── vgg16-nailclassification.ipynb
│   ├── resnet50-nail-classification (1).ipynb
│   ├── densenet-10-epoch-and-32-batch.ipynb
│   ├── GoogleNet.ipynb
│   ├── GoogleNet.html              # Exported HTML version
│   └── yolo_nail_object_detection.ipynb
├── models/                          # Model code and trained weights
│   ├── __init__.py
│   ├── disease_classifier.py       # Disease classification model
│   └── nail_detector.py            # YOLO nail detection model
├── data/                            # Dataset directory
│   ├── train/                      # Training images
│   ├── valid/                      # Validation images
│   └── test/                       # Test images
├── static/                          # Static assets for web interface
│   ├── css/
│   └── js/
├── templates/                       # HTML templates for Flask app
├── runs/                            # Training outputs
│   └── nail-detector2/             # YOLO training outputs
├── app.py                           # Flask web application
├── best.pt                          # Best YOLO model weights
├── requirements.txt                 # Python dependencies
└── MODEL_SETUP.md                   # Model setup instructions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lksimpson/NailDiseaseClassification.git
cd NailDiseaseClassification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
=======
## Features

- 🖼️ **Image Upload**: Upload hand images via web interface
- 🔍 **Nail Detection**: YOLO-based object detection to identify and crop nail regions
- 🧠 **Disease Classification**: Deep learning model to predict potential nail conditions
- 📊 **Results Display**: Probability scores and localization overlays
- ⚠️ **Medical Disclaimer**: Clear disclaimer displayed on the interface

## Architecture

### Components

1. **Backend (Flask)**: `app.py` - Main web server and API endpoints
2. **Nail Detector**: `models/nail_detector.py` - YOLO-based nail detection
3. **Disease Classifier**: `models/disease_classifier.py` - DenseNet201-based classification
4. **Frontend**: HTML/CSS/JavaScript for user interface

### Disease Classes

The model can classify the following conditions:
- Acral Lentiginous Melanoma
- Healthy Nail
- Onychogryphosis
- Blue Finger
- Clubbing
- Pitting

## Setup Instructions

### 1. Activate Virtual Environment

```bash
cd Nail-Disease
source venv/bin/activate
```

### 2. Install Dependencies

>>>>>>> feature/flask-web-application
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
## Usage

### Training Models

Each model has its own Jupyter notebook for training, located in the `notebooks/` directory:

- **AlexNet**: `notebooks/alexnet-nailclassification.ipynb`
- **VGG16**: `notebooks/vgg16-nailclassification.ipynb`
- **ResNet50**: `notebooks/resnet50-nail-classification (1).ipynb`
- **DenseNet**: `notebooks/densenet-10-epoch-and-32-batch.ipynb`
- **GoogleNet**: `notebooks/GoogleNet.ipynb`
- **YOLO Detection**: `notebooks/yolo_nail_object_detection.ipynb`

Open any notebook in Jupyter and follow the training steps to train models on your dataset. All notebooks are organized in the `notebooks/` folder for better project structure.

### Running the Web Application

#### Setup Model Weights
=======
### 3. Setup Model Weights
>>>>>>> feature/flask-web-application

**Important**: The application requires trained DenseNet model weights for accurate predictions.

**Option A: Use Trained Model**
<<<<<<< HEAD
- Train your model using `notebooks/densenet-10-epoch-and-32-batch.ipynb`
=======
- Train your model using `densenet_test_disease_classification.ipynb`
>>>>>>> feature/flask-web-application
- Save the model weights (see `MODEL_SETUP.md` for details)
- Place the `.pth` file in the `models/` directory with one of these names:
  - `densenet_nail_disease_best.pth` (recommended)
  - `densenet_nail_disease_weights.pth`
  - `densenet_model.pth`
  - `nail_disease_classifier.pth`

**Option B: Use Environment Variable**
```bash
export DISEASE_MODEL_PATH=/path/to/your/model.pth
```

**Note**: If no trained weights are found, the app will use randomly initialized weights (not suitable for predictions). The app will automatically search for model files and display a message indicating whether weights were loaded.

For YOLO: The application will automatically download YOLOv8n for nail detection.

<<<<<<< HEAD
#### Start the Application
=======
### 4. Run the Application
>>>>>>> feature/flask-web-application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

<<<<<<< HEAD
#### Using the Application
=======
## Usage
>>>>>>> feature/flask-web-application

1. Open the web interface in your browser
2. Upload a hand image (PNG, JPG, JPEG, GIF, WEBP, max 16MB)
3. Click "Analyze Image"
4. View the results:
   - Detection visualization showing detected nail regions
   - Classification results with probability scores
   - All condition probabilities for each detected nail

<<<<<<< HEAD
## Models

### Classification Models

All five classification models are trained on the same nail disease dataset:

- **AlexNet**
- **VGG16**
- **ResNet50**
- **DenseNet**
- **GoogleNet**

### Disease Classes

The models can classify the following conditions:
- Acral Lentiginous Melanoma
- Healthy Nail
- Onychogryphosis
- Blue Finger
- Clubbing
- Pitting

### Object Detection

- **YOLOv8**: Real-time nail detection to localize nail regions before classification

### Model Interpretability

- **GradCAM**: Generates heatmaps showing which regions of the image the model focuses on for its predictions

## Dataset

The models are trained on a nail disease dataset containing images of various nail conditions. 

## Model Performance

Each model notebook contains:
- Training and validation accuracy curves
- Confusion matrices
- Classification reports
- Sample predictions

Refer to individual notebooks for detailed performance metrics.

## API Endpoints

The Flask application provides the following endpoints:

=======
## Project Structure

```
Nail-Disease/
├── app.py                      # Main Flask application
├── models/
│   ├── __init__.py
│   ├── nail_detector.py        # YOLO nail detection
│   └── disease_classifier.py   # Disease classification
├── templates/
│   └── index.html              # Main web page
├── static/
│   ├── css/
│   │   └── style.css           # Styling
│   └── js/
│       └── app.js              # Frontend JavaScript
├── uploads/                     # Uploaded images (created automatically)
├── results/                     # Results storage (created automatically)
├── requirements.txt             # Python dependencies
└── README_APP.md               # This file
```

## API Endpoints

>>>>>>> feature/flask-web-application
### `GET /`
Main page with upload interface

### `GET /health`
Health check endpoint
<<<<<<< HEAD
- Returns: JSON with status and model loading information
- Example response: `{"status": "healthy", "models_loaded": true}`
=======
>>>>>>> feature/flask-web-application

### `POST /predict`
Main prediction endpoint
- **Input**: Multipart form data with `image` field
- **Output**: JSON with detection and classification results
<<<<<<< HEAD
- **Response includes**:
  - `success`: Boolean indicating if prediction was successful
  - `num_nails_detected`: Number of nails found in the image
  - `results`: Array of classification results for each detected nail
    - `nail_index`: Index of the detected nail
    - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
    - `detection_confidence`: Confidence score for nail detection
    - `disease`: Predicted disease class
    - `probability`: Confidence score for the prediction
    - `all_probabilities`: Dictionary of probabilities for all classes
  - `detection_visualization`: Base64-encoded image with bounding boxes

## Application Architecture

### Components

1. **Backend (Flask)**: `app.py` - Main web server and API endpoints
2. **Nail Detector**: `models/nail_detector.py` - YOLO-based nail detection
3. **Disease Classifier**: `models/disease_classifier.py` - DenseNet201-based classification
4. **Frontend**: HTML/CSS/JavaScript for user interface

## Development Notes
=======

## Notes
>>>>>>> feature/flask-web-application

- **Model Weights Required**: The DenseNet classifier requires trained model weights to make accurate predictions. See `MODEL_SETUP.md` for instructions on how to save and load trained models.
- The YOLO model uses a general-purpose pretrained model. For better nail detection, train a custom YOLO model on nail-specific data.
- The disease classifier uses DenseNet201 architecture. You must train the model on your dataset and save the weights for production use.
- The application is designed for screening purposes only, not medical diagnosis.

<<<<<<< HEAD
### Future Improvements
=======
## Development

To improve the application:
>>>>>>> feature/flask-web-application

1. Train a custom YOLO model for better nail detection
2. Train the DenseNet201 classifier on your dataset
3. Add model persistence and caching
4. Add user authentication and history
5. Improve error handling and validation
<<<<<<< HEAD
6. Add GradCAM visualization support
=======
>>>>>>> feature/flask-web-application
