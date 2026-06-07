from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.layers import Dense


BASE_DIR = Path(__file__).resolve().parent
DISEASE_DIR = BASE_DIR / "disease predictor"
CHATBOT_DIR = BASE_DIR / "Chatbot"
SKIN_DIR = BASE_DIR / "Skin Disease"


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def patch_dense_from_config() -> None:
    original_from_config = Dense.from_config

    @classmethod
    def patched_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_from_config.__func__(cls, config)

    Dense.from_config = patched_from_config


class DiseaseEngine:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.model = self._load_model()
        self.features = list(self.model.feature_names_in_)
        self.feature_alias_to_key, self.feature_flat_to_key = self._build_feature_lookup(self.features)
        self.label_decoder = self._load_label_decoder()
        self.metadata_by_disease = self._load_metadata_map()

    def _load_model(self):
        model_path = self.folder / "model.json"
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    def _load_label_decoder(self):
        encoder_path = self.folder / "label_encoder.pkl"
        mapping_path = self.folder / "disease_mapping.json"

        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
            return {"type": "label_encoder", "object": label_encoder}

        if mapping_path.exists():
            mapping_dict = json.loads(mapping_path.read_text(encoding="utf-8"))
            return {"type": "json_map", "object": mapping_dict}

        raise FileNotFoundError("Neither label_encoder.pkl nor disease_mapping.json found in disease predictor folder.")

    def _load_metadata_map(self) -> dict[str, dict[str, Any]]:
        preferred = self.folder / "final_disease_mapping.csv"
        fallback = self.folder / "Semifinal_health_dataset.csv"
        csv_path = preferred if preferred.exists() else fallback

        if not csv_path.exists():
            return {}

        df = pd.read_csv(csv_path)
        disease_candidates = [col for col in df.columns if "disease" in col.lower()]
        if not disease_candidates:
            return {}

        disease_col = disease_candidates[0]
        metadata_cols = [col for col in df.columns if col != disease_col]
        result: dict[str, dict[str, Any]] = {}

        for _, row in df.iterrows():
            disease_name = normalize_text(row[disease_col])
            if not disease_name:
                continue
            metadata = {
                col: safe_json_value(row[col]) for col in metadata_cols if pd.notna(row[col])
            }
            if disease_name not in result:
                result[disease_name] = metadata

        return result

    def _build_feature_lookup(self, features: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        alias_to_key: dict[str, str] = {}
        flat_to_key: dict[str, str] = {}

        for feat in features:
            display = feat.replace("_", " ")
            aliases = {
                normalize_text(feat),
                normalize_text(display),
                normalize_text(display.replace("-", " ")),
            }

            for alias in aliases:
                alias_to_key[alias] = feat
                flat_to_key[alias.replace(" ", "")] = feat

        return alias_to_key, flat_to_key

    def _decode_class_index(self, class_index: int) -> str:
        decoder_type = self.label_decoder["type"]
        decoder_obj = self.label_decoder["object"]

        if decoder_type == "label_encoder":
            return str(decoder_obj.inverse_transform([class_index])[0])

        if decoder_type == "json_map":
            return str(
                decoder_obj.get(str(class_index))
                or decoder_obj.get(class_index)
                or f"class_{class_index}"
            )

        return f"class_{class_index}"

    def _pick_best_metadata(self, disease_name: str) -> dict[str, Any]:
        normalized = normalize_text(disease_name)
        if normalized in self.metadata_by_disease:
            return self.metadata_by_disease[normalized]

        query_tokens = set(normalized.split())
        if not query_tokens:
            return {}

        best_name = None
        best_score = 0.0
        best_weight = -1.0

        for candidate_name, metadata in self.metadata_by_disease.items():
            candidate_tokens = set(candidate_name.split())
            if not candidate_tokens:
                continue

            overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
            if normalized in candidate_name or candidate_name in normalized:
                overlap += 0.35

            weight = float(metadata.get("Prior_Weight", 0) or 0)
            if overlap > best_score or (abs(overlap - best_score) < 1e-9 and weight > best_weight):
                best_score = overlap
                best_weight = weight
                best_name = candidate_name

        if best_name and best_score >= 0.5:
            return self.metadata_by_disease[best_name]
        return {}

    def _format_timeline(self, metadata: dict[str, Any]) -> str:
        min_days = metadata.get("Typical_Min_Duration_Days")
        max_days = metadata.get("Typical_Max_Duration_Days")
        if min_days is None and max_days is None:
            return "Unknown"
        if min_days is not None and max_days is not None:
            if int(min_days) == int(max_days):
                return f"{int(max_days)} days"
            return f"{int(min_days)}-{int(max_days)} days"
        if max_days is not None:
            return f"{int(max_days)} days"
        return f"{int(min_days)} days"

    def _recommended_action(self, urgency: str) -> str:
        level = normalize_text(urgency)
        if level in {"low", "routine"}:
            return "ROUTINE: Monitor symptoms and rest; consult a doctor if condition persists."
        if level in {"moderate", "medium"}:
            return "SOON: Book a doctor consultation within 24-48 hours."
        if level in {"high", "urgent", "emergency"}:
            return "URGENT: Seek immediate medical care."
        return "Consult a doctor for clinical confirmation and next steps."

    def get_dropdown_symptoms(self) -> list[dict[str, str]]:
        return [{"key": feat, "label": feat.replace("_", " ").title()} for feat in self.features]

    def _map_input_symptoms(self, symptoms: list[str]) -> tuple[list[str], list[str]]:
        matched: list[str] = []
        unknown: list[str] = []
        used_keys: set[str] = set()

        for symptom in symptoms:
            alias = normalize_text(symptom).replace("-", " ")
            candidate = self.feature_alias_to_key.get(alias)
            if not candidate:
                candidate = self.feature_flat_to_key.get(alias.replace(" ", ""))

            if candidate and candidate not in used_keys:
                used_keys.add(candidate)
                matched.append(candidate)
            elif not candidate:
                unknown.append(str(symptom))

        return matched, unknown

    def predict(self, symptoms: list[str]) -> dict[str, Any]:
        matched, unknown = self._map_input_symptoms(symptoms)
        if not matched:
            raise ValueError("No valid symptoms matched model feature list.")

        vector = np.zeros(len(self.features), dtype=np.float32)
        feature_index = {feature: idx for idx, feature in enumerate(self.features)}
        for symptom_key in matched:
            vector[feature_index[symptom_key]] = 1.0

        probabilities = self.model.predict_proba(vector.reshape(1, -1))[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions = []
        for index in top_indices:
            disease_name = self._decode_class_index(int(index))
            metadata = self._pick_best_metadata(disease_name)

            top_predictions.append(
                {
                    "disease": disease_name,
                    "confidence_percent": round(float(probabilities[index]) * 100, 4),
                    "urgency": metadata.get("Urgency_Level", "Unknown"),
                    "organ_system": metadata.get("Primary_Organ_System", "Unknown"),
                    "timeline": self._format_timeline(metadata),
                    "likelihood": metadata.get("Prevalence", "Unknown"),
                    "recommended_action": self._recommended_action(metadata.get("Urgency_Level", "Unknown")),
                }
            )

        primary = top_predictions[0]
        return {
            "used_symptoms": [symptom.replace("_", " ") for symptom in matched],
            "unmatched_symptoms": unknown,
            "primary_prediction": primary,
            "top_predictions": top_predictions,
        }


class ChatbotEngine:
    def __init__(self, folder: Path) -> None:
        model, features = joblib.load(folder / "disease_model.pkl")
        self.model = model
        self.features = [str(f).strip() for f in features]
        self.feature_set = set(self.features)
        self.description_map = self._load_description_map(folder)
        self.precaution_map = self._load_precaution_map(folder)
        self.emergency_symptoms = {
            "chest_pain",
            "breathlessness",
            "blood_in_sputum",
            "fast_heart_rate",
            "yellowing_of_eyes",
            "vomiting",
            "blackouts",
            "altered_sensorium",
        }

    def _load_description_map(self, folder: Path) -> dict[str, str]:
        candidates = [
            folder / "data" / "symptom_Description.csv",
            folder / "symptom_Description.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if {"Disease", "Description"}.issubset(df.columns):
                    return {
                        normalize_text(row["Disease"]): str(row["Description"]).strip()
                        for _, row in df.iterrows()
                        if pd.notna(row["Disease"]) and pd.notna(row["Description"])
                    }
            except Exception:
                continue
        return {}

    def _load_precaution_map(self, folder: Path) -> dict[str, list[str]]:
        candidates = [
            folder / "data" / "symptom_precaution.csv",
            folder / "symptom_precaution.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                disease_col = next((c for c in df.columns if normalize_text(c) == "disease"), None)
                precaution_cols = [c for c in df.columns if "precaution" in normalize_text(c)]
                if not disease_col or not precaution_cols:
                    continue
                result: dict[str, list[str]] = {}
                for _, row in df.iterrows():
                    disease_name = normalize_text(row[disease_col])
                    if not disease_name:
                        continue
                    precautions = [
                        str(row[col]).strip()
                        for col in precaution_cols
                        if pd.notna(row[col]) and str(row[col]).strip()
                    ]
                    if precautions:
                        result[disease_name] = precautions
                return result
            except Exception:
                continue
        return {}

    def _extract_symptoms(self, message: str) -> list[str]:
        normalized_message = f" {normalize_text(re.sub(r'[^a-zA-Z0-9_ ]+', ' ', message))} "
        matched = []
        for symptom in self.features:
            phrase = symptom.replace("_", " ")
            if f" {phrase} " in normalized_message:
                matched.append(symptom)
        return list(dict.fromkeys(matched))

    def _symptom_weight(self, symptom: str) -> int:
        high_keys = {
            "chest_pain",
            "breathlessness",
            "blood_in_sputum",
            "altered_sensorium",
            "vomiting",
            "high_fever",
            "severe",
            "unconscious",
        }
        medium_keys = {
            "fever",
            "headache",
            "fatigue",
            "dizziness",
            "abdominal_pain",
            "joint_pain",
            "nausea",
            "yellowing",
        }
        if any(key in symptom for key in high_keys):
            return 5
        if any(key in symptom for key in medium_keys):
            return 3
        return 2

    def _severity_score(self, symptoms: list[str]) -> int:
        return int(sum(self._symptom_weight(symptom) for symptom in symptoms))

    def _risk_level(self, score: int, emergency: bool) -> str:
        if emergency or score >= 15:
            return "High"
        if score >= 8:
            return "Medium"
        return "Low"

    def _detect_emergency(self, symptoms: list[str], message: str) -> bool:
        if any(symptom in self.emergency_symptoms for symptom in symptoms):
            return True
        urgent_words = {"severe bleeding", "unconscious", "cannot breathe", "breathlessness", "chest pain"}
        normalized_message = normalize_text(message)
        return any(phrase in normalized_message for phrase in urgent_words)

    def _follow_up_questions(self, symptoms: list[str], risk: str, emergency: bool) -> list[str]:
        symptom_set = set(symptoms)
        questions: list[str] = []

        if "fever" in " ".join(symptoms):
            questions.append("Do you also have chills or sweating?")
        if any("cough" in s for s in symptoms):
            questions.append("Is your cough dry or with mucus/blood?")
        if any("chest_pain" in s for s in symptoms):
            questions.append("Does chest pain increase while breathing or moving?")
        if any("skin_rash" in s or "itching" in s for s in symptoms):
            questions.append("Has the rash spread to new areas recently?")
        if "nausea" in symptom_set or "vomiting" in symptom_set:
            questions.append("Are you able to keep food and fluids down?")

        if risk != "Low":
            questions.append("How long have these symptoms persisted?")
            questions.append("Have symptoms become worse in the last 24 hours?")

        if emergency:
            questions.append("Are you currently alone, or is someone with you for immediate help?")

        if not questions:
            questions = [
                "When did your symptoms begin?",
                "Have you taken any medication already?",
                "Do you have any known chronic conditions?",
            ]

        return questions[:4]

    def _description(self, disease: str) -> str:
        description = self.description_map.get(normalize_text(disease))
        if description:
            return description
        return (
            f"{disease} is a possible condition based on the currently matched symptoms. "
            "Please use this as preliminary guidance only."
        )

    def _precautions(self, disease: str, risk: str, emergency: bool) -> list[str]:
        mapped = self.precaution_map.get(normalize_text(disease))
        if mapped:
            return mapped[:4]

        if emergency:
            return [
                "Seek emergency medical care immediately.",
                "Do not ignore worsening symptoms.",
                "Avoid self-medication until examined by a doctor.",
            ]
        if risk == "High":
            return [
                "Arrange urgent doctor consultation today.",
                "Rest and avoid physical exertion.",
                "Track symptom progression every few hours.",
            ]
        if risk == "Medium":
            return [
                "Book a doctor consultation within 24-48 hours.",
                "Stay hydrated and maintain proper rest.",
                "Monitor symptoms and seek help if worsening.",
            ]
        return [
            "Monitor symptoms and rest.",
            "Maintain hydration and healthy diet.",
            "Consult a doctor if condition persists.",
        ]

    def respond(self, message: str, session_symptoms: list[str] | None = None) -> dict[str, Any]:
        current_symptoms = self._extract_symptoms(message)
        existing_session = [normalize_text(s).replace(" ", "_") for s in (session_symptoms or [])]
        existing_session = [s for s in existing_session if s in self.feature_set]
        normalized_message = normalize_text(message)

        explicit_new_chat_markers = {
            "new symptom",
            "new symptoms",
            "different symptom",
            "different symptoms",
            "start new chat",
            "reset chat",
            "fresh start",
        }
        message_requests_reset = any(marker in normalized_message for marker in explicit_new_chat_markers)

        session_reset = False
        if message_requests_reset and existing_session:
            existing_session = []
            session_reset = True

        if current_symptoms and existing_session:
            current_set = set(current_symptoms)
            existing_set = set(existing_session)
            has_new_symptom = len(current_set - existing_set) > 0
            if has_new_symptom or message_requests_reset:
                existing_session = []
                session_reset = True

        combined_symptoms = list(dict.fromkeys(existing_session + current_symptoms))
        if not combined_symptoms:
            return {
                "Possible_Condition": "Insufficient symptom match",
                "Matched_Symptoms": [],
                "Description": "I could not map enough known symptoms from your text.",
                "Recommended_Precautions": [
                    "Please provide clear symptom names (for example: fever, cough, chest pain, headache).",
                    "If symptoms are severe, consult a doctor immediately.",
                ],
                "Risk_Level": "Unknown",
                "Severity_Score": 0,
                "Confidence_percent": 0.0,
                "Top_3_Predictions": [],
                "Follow_up_Questions": [
                    "How long have you had these symptoms?",
                    "Do you have fever, cough, or chest pain?",
                ],
                "Session_Symptoms": [],
                "Session_Reset": bool(message_requests_reset),
                "Emergency": False,
                "Disclaimer": "This is not a medical diagnosis. Please consult a qualified doctor.",
            }

        vector = np.zeros(len(self.features), dtype=np.float32)
        index_map = {feature: idx for idx, feature in enumerate(self.features)}
        for symptom in combined_symptoms:
            vector[index_map[symptom]] = 1.0

        probabilities = self.model.predict_proba(vector.reshape(1, -1))[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "Disease": str(self.model.classes_[int(idx)]).strip(),
                "Probability_percent": round(min(float(probabilities[int(idx)]) * 100, 95.0), 2),
            }
            for idx in top_indices
        ]

        top_1 = top_predictions[0]
        disease = top_1["Disease"]
        confidence = top_1["Probability_percent"]
        severity_score = self._severity_score(combined_symptoms)
        emergency = self._detect_emergency(combined_symptoms, message)
        risk_level = self._risk_level(severity_score, emergency)
        description = self._description(disease)
        precautions = self._precautions(disease, risk_level, emergency)
        follow_ups = self._follow_up_questions(combined_symptoms, risk_level, emergency)

        return {
            "Possible_Condition": disease,
            "Matched_Symptoms": [symptom.replace("_", " ").title() for symptom in combined_symptoms],
            "Description": description,
            "Recommended_Precautions": precautions,
            "Risk_Level": risk_level,
            "Severity_Score": severity_score,
            "Confidence_percent": confidence,
            "Top_3_Predictions": top_predictions,
            "Follow_up_Questions": follow_ups,
            "Session_Symptoms": combined_symptoms,
            "Session_Reset": session_reset,
            "Emergency": emergency,
            "Disclaimer": "This is not a medical diagnosis. Please consult a qualified doctor.",
        }


class SkinDiseaseEngine:
    def __init__(self, folder: Path) -> None:
        patch_dense_from_config()
        self.model = joblib.load(folder / "skin_disease_model.pkl")
        self.class_names = self._load_class_names(folder, output_size=self.model.output_shape[-1])

    def _load_class_names(self, folder: Path, output_size: int) -> list[str]:
        json_candidates = [folder / "class_names.json", folder / "labels.json"]
        txt_candidates = [folder / "class_names.txt", folder / "labels.txt"]

        for path in json_candidates:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    ordered = [raw[str(i)] if str(i) in raw else raw.get(i, f"Skin Class {i + 1}") for i in range(output_size)]
                    return [str(x) for x in ordered]
                if isinstance(raw, list) and len(raw) >= output_size:
                    return [str(x) for x in raw[:output_size]]

        for path in txt_candidates:
            if path.exists():
                labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(labels) >= output_size:
                    return labels[:output_size]

        if output_size == 7:
            # Common 7-class dermatology mapping fallback (editable via class_names.json).
            return [
                "Actinic Keratoses",
                "Basal Cell Carcinoma",
                "Benign Keratosis-like Lesions",
                "Dermatofibroma",
                "Melanoma",
                "Melanocytic Nevi",
                "Vascular Lesions",
            ]

        return [f"Skin Condition {i + 1}" for i in range(output_size)]

    def predict(self, image_file) -> dict[str, Any]:
        image = Image.open(image_file.stream).convert("RGB").resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = self.model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "class_index": int(idx),
                "class_name": self.class_names[int(idx)],
                "confidence_percent": round(float(probabilities[idx]) * 100, 4),
            }
            for idx in top_indices
        ]

        top = top_predictions[0]
        return {
            "predicted_index": top["class_index"],
            "predicted_class": top["class_name"],
            "confidence_percent": top["confidence_percent"],
            "top_predictions": top_predictions,
        }


disease_engine = DiseaseEngine(DISEASE_DIR)
chatbot_engine = ChatbotEngine(CHATBOT_DIR)
skin_engine = SkinDiseaseEngine(SKIN_DIR)

app = Flask(__name__)
CORS(app)
CHAT_SESSIONS: dict[str, dict[str, Any]] = {}


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/style.css")
def css_file():
    return send_from_directory(BASE_DIR, "style.css")


@app.get("/script.js")
def js_file():
    return send_from_directory(BASE_DIR, "script.js")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/symptoms")
def list_symptoms():
    return jsonify({"symptoms": disease_engine.get_dropdown_symptoms()})


@app.post("/api/predict")
def predict_disease():
    try:
        payload = request.get_json(silent=True) or {}
        symptoms = payload.get("symptoms", [])
        if not isinstance(symptoms, list):
            return jsonify({"error": "symptoms must be a list."}), 400

        result = disease_engine.predict([str(x) for x in symptoms])
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.post("/api/chat")
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        session_id = str(payload.get("session_id", "")).strip()
        session_symptoms = payload.get("session_symptoms", [])
        if not message:
            return jsonify({"error": "message is required."}), 400
        if not isinstance(session_symptoms, list):
            return jsonify({"error": "session_symptoms must be a list."}), 400
        if not session_id:
            session_id = str(uuid.uuid4())

        memory = CHAT_SESSIONS.get(session_id, {"session_symptoms": []})
        merged_session_symptoms = session_symptoms or memory.get("session_symptoms", [])

        response_data = chatbot_engine.respond(message, session_symptoms=merged_session_symptoms)
        if response_data.get("Session_Reset"):
            session_id = str(uuid.uuid4())
        CHAT_SESSIONS[session_id] = {
            "session_symptoms": response_data.get("Session_Symptoms", []),
        }
        response_data["session_id"] = session_id
        return jsonify(response_data)
    except Exception as exc:
        return jsonify({"error": f"Chat failed: {exc}"}), 500


@app.post("/api/skin-predict")
def predict_skin():
    try:
        if "image" not in request.files:
            return jsonify({"error": "image file is required in form-data."}), 400

        image = request.files["image"]
        if image.filename == "":
            return jsonify({"error": "image file name is empty."}), 400

        result = skin_engine.predict(image)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Skin prediction failed: {exc}"}), 500


@app.post("/api/book-appointment")
def book_appointment():
    payload = request.get_json(silent=True) or {}
    required_fields = ["full_name", "email", "date", "time", "department", "concern"]
    missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    return jsonify(
        {
            "message": "Appointment request received successfully.",
            "note": (
                "Next step: connect your email/calendar API in this endpoint "
                "to send Meet link directly to patient email."
            ),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
