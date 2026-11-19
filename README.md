# Plant Disease Detection

An AI-powered web application that analyzes plant leaf images and predicts potential diseases using a Convolutional Neural Network (CNN).

The application is built with **TensorFlow** and **Streamlit** and supports **38 plant disease classes** from the PlantVillage dataset.

## Project Overview

Plant diseases can affect crop health and agricultural productivity. This project demonstrates how deep learning can be used to classify plant leaf images and provide a quick model-generated prediction.

Users can upload a leaf image through the web interface. The trained CNN processes the image and returns the predicted class along with the model's confidence score.

## Highlights

- CNN-based image classification.
- 38 supported plant and disease classes.
- Image upload through a Streamlit interface.
- Prediction confidence display.
- Dark, agriculture-inspired user interface.
- Model loading with Streamlit resource caching.

## Model Performance

Architecture => Convolutional Neural Network (CNN) 
Number of supported classes => 38 
Test accuracy => 95.98% 
Input image size => 224 × 224 pixels

The reported accuracy is the result obtained during evaluation on this project's test set. It should not be interpreted as a guarantee of the same performance on every real-world image.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Streamlit

## Project Structure


Plant-Disease-Detection/
│
├── app/
│   └── app.py
│
├── models/
│   ├── plant_disease_model.keras
│   └── class_names.json
│
├── requirements.txt
├── .gitignore
└── README.md


## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Plant-Disease-Detection
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your GitHub repository.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

From the project root directory:

```bash
streamlit run app/app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

## How to Use

1. Start the Streamlit application.
2. Upload a plant leaf image in JPG, JPEG, or PNG format.
3. Select **Analyze Leaf Image**.
4. View the predicted class and confidence score.

For best results, use a clear, well-lit image in which the leaf is visible.

## Important Limitations

- The model predicts only among the 38 classes represented in its training labels.
- It may produce an incorrect prediction for images outside those classes.
- A high confidence score does not guarantee that the prediction is correct.
- Performance on real-world photographs may differ from performance on the test dataset.
- This application is an educational and experimental AI tool. It is not a substitute for professional agricultural diagnosis.

## Future Scope

Possible future improvements include:

- Testing with more diverse real-world leaf images.
- Adding additional crops and disease categories.
- Providing verified disease information and management guidance.
- Deploying the application for public access.

## Author

**Sayantan Metya**

- GitHub: [@SR4-ai](https://github.com/SR4-ai)

Developed as a deep learning project using CNN, TensorFlow, and Streamlit.

