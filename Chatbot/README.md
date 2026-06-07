🏥 Advanced AI Medical Chatbot Backend
An intelligent, Machine Learning–powered medical chatbot API built using FastAPI.
The system predicts possible diseases from user symptoms, calculates severity scores, detects emergencies, and provides structured medical guidance for frontend integration.

📌 Project Overview
This backend system is designed as part of an AI Healthcare Dashboard project.

The chatbot:

Parses free-text symptom input

Predicts disease using a trained RandomForest model

Calculates severity score using weighted symptom data

Classifies risk level (Low / Medium / High)

Detects emergency symptoms

Suggests follow-up questions

Returns structured JSON response

Provides confidence percentage & top-3 predictions

🚀 Key Features
✅ ML-based disease classification (RandomForest)

✅ NLP-based symptom extraction

✅ Symptom severity scoring system

✅ Risk level classification

✅ Emergency detection logic

✅ Top-3 probability distribution

✅ Realistic confidence capping

✅ Follow-up question generation

✅ Structured API response format

✅ CORS enabled for frontend integration

✅ Health check endpoint

✅ Production-ready FastAPI backend

🧠 System Workflow
User Input (Free Text Symptoms)
↓
Symptom Parser (Text → Known Symptoms)
↓
ML Classifier (Binary Symptom Vector → Disease Prediction)
↓
Severity Engine (Weighted Score Calculation)
↓
Risk Classification
↓
Emergency Detection
↓
Response Formatter
↓
Structured JSON Output

📂 Project Structure
medical_chatbot_v2/
│
├── data/
│   ├── dataset.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
│
├── models/
│   └── disease_model.pkl
│
├── services/
│   ├── ml_predictor.py
│   ├── symptom_parser.py
│   ├── severity_engine.py
│   ├── emergency.py
│   ├── followup_engine.py
│   ├── formatter.py
│
├── train_model.py
├── main.py
├── requirements.txt
└── README.md
⚙️ Installation Guide
1️⃣ Clone or Download the Project
git clone <your-repository-url>
cd medical_chatbot_v2
Or download ZIP and extract.

2️⃣ Create Virtual Environment
Windows:

python -m venv venv
venv\Scripts\activate
Mac/Linux:

python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Train the ML Model (Run Once)
Before running the API, generate the model file:

python train_model.py
This creates:

models/disease_model.pkl
5️⃣ Run the Backend Server
uvicorn main:app --reload
Server runs at:

http://127.0.0.1:8000
📡 API Endpoints
🔹 Health Check
GET /

http://127.0.0.1:8000/
Response:

{
  "status": "Medical Chatbot API is running"
}
🔹 Chat Endpoint
POST /chat

http://127.0.0.1:8000/chat
📥 Request Format
{
  "message": "I have fever, headache and chest pain"
}
📤 Response Format
{
  "Possible_Condition": "Disease Name",
  "Matched_Symptoms": ["Fever", "Chest pain"],
  "Description": "Disease explanation...",
  "Recommended_Precautions": ["Rest", "Consult doctor"],
  "Risk_Level": "High",
  "Severity_Score": 18,
  "Confidence_percent": 92.5,
  "Top_3_Predictions": [
    {"Disease": "Disease A", "Probability_percent": 92.5},
    {"Disease": "Disease B", "Probability_percent": 4.1},
    {"Disease": "Disease C", "Probability_percent": 2.3}
  ],
  "Follow_up_Questions": ["Do you have nausea?"],
  "Emergency": false,
  "Disclaimer": "This is not a medical diagnosis. Please consult a qualified doctor."
}
🚨 Emergency Detection
The system automatically flags emergency symptoms such as:

Chest pain

Breathlessness

Unconsciousness

Severe bleeding

If detected:

"Emergency": true
The system immediately advises seeking medical help.

📊 Machine Learning Details
Algorithm: RandomForestClassifier

Feature Encoding: Binary symptom vector

Output: Disease prediction + probability distribution

Confidence capped at 95% for realistic medical output

Deterministic dataset-driven predictions (no hallucination)

🛡 Safety & Ethics
No prescription-level advice given

Medical disclaimer included in every response

Emergency override logic implemented

Designed for educational & clinical support use

Not a substitute for professional medical consultation

🔗 Frontend Integration Guide
Frontend developers should use:

POST http://localhost:8000/chat
Headers:

Content-Type: application/json
CORS is already enabled.

🧪 Example High Severity Test Input
{
  "message": "I have chest pain, breathlessness, high fever and severe headache"
}
