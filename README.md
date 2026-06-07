# MedPredict - AI-Powered Disease Prediction and Healthcare Assistant

## Overview

MedPredict is an intelligent healthcare platform that combines Machine Learning, Deep Learning, Natural Language Processing (NLP), and Medical Decision Support Systems to provide preliminary health assessments.

The platform integrates multiple healthcare services into a single ecosystem, enabling users to:

* Predict diseases based on symptoms
* Analyze skin conditions using image classification
* Interact with an AI-powered medical chatbot
* Book consultations with healthcare professionals

MedPredict aims to bridge the gap between healthcare accessibility and timely medical guidance, particularly for users in remote and underserved regions.

---

# Problem Statement

Millions of people lack immediate access to healthcare professionals, leading to:

* Delayed diagnosis
* Self-diagnosis errors
* Lack of specialist access in rural regions
* Increased healthcare costs
* Delayed medical intervention

Existing healthcare applications often focus on a single domain such as symptom checking or dermatological analysis, forcing users to rely on multiple disconnected systems.

MedPredict addresses these challenges by providing a unified AI-powered healthcare ecosystem capable of handling both textual and visual medical inputs.

---

# Objectives

* Improve early disease detection
* Support rural healthcare accessibility
* Promote public health awareness
* Encourage timely medical consultation
* Reduce self-diagnosis errors
* Provide actionable healthcare guidance

---

# Key Features

## Symptom-Based Disease Prediction

Predicts diseases from user-entered symptoms using Machine Learning models.

Features:

* NLP-based symptom extraction
* Disease prediction across 150+ conditions
* Confidence score generation
* Top-3 probable disease predictions
* Severity-based risk analysis

---

## Skin Disease Detection

Users can upload images of skin lesions for preliminary screening.

Features:

* Image-based disease classification
* MobileNetV2 transfer learning model
* Top predictions with confidence scores
* Fast inference (< 2 seconds)

---

## AI Medical Chatbot

An intelligent healthcare assistant capable of:

* Symptom interpretation
* Disease prediction
* Risk assessment
* Follow-up questioning
* Medical precautions and recommendations

---

## Severity-Based Risk Assessment

Cases are classified into:

* Low Risk
* Medium Risk
* High Risk

Emergency symptoms trigger additional warnings and precautionary recommendations.

---

## Appointment Booking Module

Allows users to:

* Book consultations
* Schedule appointments
* Generate meeting links
* Receive confirmations

---

# System Architecture

```text
┌─────────────────────────────────────────────┐
│               Frontend Layer                │
│ HTML • CSS • JavaScript                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│               FastAPI Backend               │
│ API Routing & Business Logic                │
└─────────────────┬───────────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼

 Disease      Skin Disease    Chatbot
Prediction    Prediction      Engine

(XGBoost)     (MobileNetV2)   (NLP)

      │           │           │
      └───────────┼───────────┘
                  │
                  ▼

          Risk Assessment
          & Confidence Layer

                  │
                  ▼

            User Response
```

---

# Technology Stack

## Programming Languages

* Python
* JavaScript
* HTML5
* CSS3

---

## Backend

* FastAPI
* Uvicorn

---

## Machine Learning

* XGBoost
* Random Forest
* Scikit-Learn

---

## Deep Learning

* TensorFlow
* Keras
* MobileNetV2

---

## Data Processing

* NumPy
* Pandas

---

## Computer Vision

* OpenCV
* PIL

---

## Development Tools

* VS Code
* PyCharm
* Google Colab
* Jupyter Notebook
* Postman

---

# Datasets

## Symptom Prediction Dataset

* 18,000+ records
* 131 unique symptoms
* 150 disease categories

Used for:

* Symptom-to-disease mapping
* Severity analysis
* Risk prediction

---

## HAM10000 Dataset

Human Against Machine Dataset

Contains:

* 10,015 dermatoscopic images
* 7 diagnostic skin lesion classes

Used for:

* Skin disease classification
* Transfer learning training

---

# Machine Learning Pipeline

## Step 1: Symptom Processing

User enters symptoms in natural language.

Example:

```text
I have a headache, fever and body pain.
```

The NLP engine extracts relevant symptoms and converts them into structured features.

---

## Step 2: Feature Engineering

* Symptom cleaning
* One-Hot Encoding
* Binary feature vector generation

131 symptoms are transformed into machine-readable vectors.

---

## Step 3: Disease Prediction

Models Evaluated:

* Random Forest
* XGBoost

Final Production Model:

* XGBoost

Reasons:

* Better sparse-data handling
* Faster inference
* Improved scalability

---

## Step 4: Severity Analysis

A severity engine calculates:

* Severity score
* Risk level
* Emergency indicators

Risk Levels:

```text
Low Risk
Medium Risk
High Risk
```

---

## Step 5: Response Generation

Outputs:

* Predicted disease
* Confidence score
* Risk level
* Precautions
* Follow-up recommendations

---

# Skin Disease Prediction Pipeline

## Image Input

User uploads:

```text
JPG
PNG
JPEG
```

---

## Preprocessing

* Image resizing (224 × 224)
* Normalization
* Augmentation

---

## Deep Learning Model

### MobileNetV2

Transfer Learning Approach:

* Pre-trained ImageNet weights
* Fine-tuned on HAM10000 dataset
* Custom classification layers
* Dropout regularization (30%)

---

## Output

Provides:

* Predicted skin condition
* Confidence score
* Secondary possible diagnoses

---

# Model Performance

## Disease Prediction Model

| Metric    | XGBoost |
| --------- | ------- |
| Accuracy  | 66.25%  |
| Precision | 67.11%  |
| Recall    | 65.93%  |
| F1 Score  | 64.60%  |

---

## Skin Disease Model

| Metric            | Performance |
| ----------------- | ----------- |
| Training Accuracy | ~90%        |
| Inference Time    | < 2 Seconds |
| Classes           | 7           |

---

## System Performance

| Metric               | Value    |
| -------------------- | -------- |
| Diseases Supported   | 150+     |
| Symptoms Supported   | 131      |
| API Response Time    | < 800 ms |
| Skin Prediction Time | < 2 sec  |

---

# Safety Features

## Confidence Filtering

Predictions below 40% confidence trigger:

```text
Low Confidence Warning
```

This prevents unreliable automated recommendations.

---

## Emergency Symptom Detection

Flags critical symptoms such as:

* Chest Pain
* Breathing Difficulty
* Severe Neurological Symptoms

Users are advised to seek immediate medical attention when necessary.

---

# Project Structure

```text
MedPredict/
│
├── frontend/
│   ├── html/
│   ├── css/
│   └── js/
│
├── backend/
│   ├── api/
│   ├── models/
│   └── services/
│
├── datasets/
│
├── disease_prediction/
│
├── skin_prediction/
│
├── chatbot/
│
├── appointment_booking/
│
├── trained_models/
│
├── notebooks/
│
├── requirements.txt
│
├── main.py
│
└── README.md
```

---

# Use Cases

## Digital Healthcare Assistant

Provides preliminary health guidance before a hospital visit.

---

## Rural Healthcare Support

Brings AI-assisted diagnostics to underserved regions.

---

## Dermatology Screening

Allows early identification of skin abnormalities.

---

## Clinical Workflow Assistance

Provides structured patient summaries before consultation.

---

## Personal Health Monitoring

Tracks recurring symptoms and risk levels over time.

---

# Future Enhancements

* Integration with Large Language Models (LLMs)
* Electronic Health Record (EHR) Support
* Multi-language Healthcare Assistance
* Wearable Device Integration
* Voice-Based Symptom Reporting
* Telemedicine Integration
* Personalized Health Recommendations

---

# Research Contributions

This project combines:

* Natural Language Processing
* Machine Learning
* Deep Learning
* Computer Vision
* Medical Risk Assessment
* Healthcare Automation

into a unified healthcare ecosystem capable of providing accessible and scalable preliminary medical guidance.

---

# Team

* Akshat Singh (24BCE10004)
* Parth Khare (24BCE10390)
* Sejal Mishra (24BCE11199)
* Vaidehi Gupta (24BCE10616)
* Divyansh Varshney (24BCE11063)

**Supervisor:** Dr. Ajeet Singh

---

# Disclaimer

MedPredict is intended for educational and preliminary screening purposes only.

The platform does not replace professional medical diagnosis, treatment, or consultation. Users should always consult qualified healthcare professionals for medical decisions.
