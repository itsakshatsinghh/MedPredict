const API_ENDPOINTS = {
  symptomList: "/api/symptoms",
  diseasePredict: "/api/predict",
  skinPredict: "/api/skin-predict",
  chatbot: "/api/chat",
  booking: "/api/book-appointment"
};

const symptomDropdown = document.getElementById("symptomDropdown");
const addSymptomBtn = document.getElementById("addSymptomBtn");
const selectedSymptomsContainer = document.getElementById("selectedSymptomsContainer");
const diseaseResult = document.getElementById("diseaseResult");
const predictBtn = document.getElementById("predictBtn");
const clearSymptomsBtn = document.getElementById("clearSymptomsBtn");

const selectedSymptoms = new Map();

function setupSmoothButtons() {
  const scrollButtons = document.querySelectorAll("[data-scroll]");
  scrollButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetSelector = button.getAttribute("data-scroll");
      const target = document.querySelector(targetSelector);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function renderSelectedSymptoms() {
  selectedSymptomsContainer.innerHTML = "";

  if (selectedSymptoms.size === 0) {
    const emptyNote = document.createElement("p");
    emptyNote.className = "small-text";
    emptyNote.textContent = "No symptom selected yet.";
    selectedSymptomsContainer.appendChild(emptyNote);
    return;
  }

  selectedSymptoms.forEach((label, key) => {
    const chip = document.createElement("span");
    chip.className = "chip active";
    chip.innerHTML = `
      ${label}
      <button class="remove-chip" type="button" aria-label="Remove symptom">&times;</button>
    `;

    chip.querySelector(".remove-chip").addEventListener("click", () => {
      selectedSymptoms.delete(key);
      renderSelectedSymptoms();
    });

    selectedSymptomsContainer.appendChild(chip);
  });
}

async function loadSymptomsIntoDropdown() {
  symptomDropdown.innerHTML = `<option>Loading symptoms...</option>`;

  try {
    const data = await apiRequest(API_ENDPOINTS.symptomList);
    symptomDropdown.innerHTML = "";

    data.symptoms.forEach((symptom, index) => {
      const option = document.createElement("option");
      option.value = symptom.key;
      option.textContent = symptom.label;
      if (index === 0) option.selected = true;
      symptomDropdown.appendChild(option);
    });
  } catch (error) {
    symptomDropdown.innerHTML = `<option>Unable to load symptoms</option>`;
    diseaseResult.innerHTML = `
      <h3>Could not load symptom list</h3>
      <p>${error.message}</p>
    `;
  }
}

function addSelectedSymptom() {
  const selectedOption = symptomDropdown.options[symptomDropdown.selectedIndex];
  if (!selectedOption) return;

  const key = selectedOption.value;
  const label = selectedOption.textContent;
  selectedSymptoms.set(key, label);
  renderSelectedSymptoms();
}

async function runPrediction() {
  if (selectedSymptoms.size === 0) {
    diseaseResult.innerHTML = `
      <h3>No symptom selected</h3>
      <p>Please add at least one symptom from the dropdown.</p>
    `;
    return;
  }

  diseaseResult.innerHTML = `
    <h3>Running model inference...</h3>
    <p>Please wait while prediction is generated.</p>
  `;

  try {
    const payload = { symptoms: [...selectedSymptoms.keys()] };
    const data = await apiRequest(API_ENDPOINTS.diseasePredict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const primary = data.primary_prediction;
    const remaining = (data.top_predictions || []).slice(1);
    const remainingHtml = remaining
      .map((item, idx) => `<li>${idx + 2}. ${item.disease} (${item.confidence_percent.toFixed(2)}%)</li>`)
      .join("");

    diseaseResult.innerHTML = `
      <h3>PRIMARY PREDICTION: ${primary.disease} (Confidence: ${primary.confidence_percent.toFixed(2)}%)</h3>
      <p><strong>Urgency:</strong> ${primary.urgency}</p>
      <p><strong>Organ System:</strong> ${primary.organ_system}</p>
      <p><strong>Timeline:</strong> ${primary.timeline}</p>
      <p><strong>Likelihood:</strong> ${primary.likelihood}</p>
      <p><strong>RECOMMENDED ACTION:</strong> ${primary.recommended_action}</p>
      ${remainingHtml ? `<p style="margin-top:8px;"><strong>Other possible predictions:</strong></p><ul style="margin:6px 0 0 16px;color:#5b6472;">${remainingHtml}</ul>` : ""}
      <p style="margin-top:8px;"><strong>Symptoms used:</strong> ${data.used_symptoms.join(", ")}</p>
    `;
  } catch (error) {
    diseaseResult.innerHTML = `
      <h3>Prediction failed</h3>
      <p>${error.message}</p>
    `;
  }
}

function clearSymptoms() {
  selectedSymptoms.clear();
  renderSelectedSymptoms();
  diseaseResult.innerHTML = `
    <h3>Prediction output will appear here</h3>
    <p>Integration point: connect this section to your disease prediction API/model.</p>
  `;
}

const skinFileInput = document.getElementById("skinFileInput");
const previewArea = document.getElementById("previewArea");
const skinResult = document.getElementById("skinResult");

async function showSkinPreview(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function handleLoad(e) {
    previewArea.innerHTML = `<img src="${e.target.result}" alt="Uploaded skin preview" />`;
  };
  reader.readAsDataURL(file);

  skinResult.innerHTML = `
    <h3>Running skin model...</h3>
    <p>Please wait while image is being analyzed.</p>
  `;

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch(API_ENDPOINTS.skinPredict, {
      method: "POST",
      body: formData
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Skin prediction failed");

    const top3 = (data.top_predictions || [])
      .map((item, idx) => `<li>${idx + 1}. ${item.class_name} (${item.confidence_percent.toFixed(2)}%)</li>`)
      .join("");

    skinResult.innerHTML = `
      <h3>Top result: ${data.predicted_class} (${data.confidence_percent.toFixed(2)}%)</h3>
      <ul style="margin:6px 0 0 16px;color:#5b6472;">${top3}</ul>
    `;
  } catch (error) {
    skinResult.innerHTML = `
      <h3>Skin prediction failed</h3>
      <p>${error.message}</p>
    `;
  }
}

const chatBox = document.getElementById("chatBox");
const chatInput = document.getElementById("chatInput");
const sendChatBtn = document.getElementById("sendChatBtn");
const newChatBtn = document.getElementById("newChatBtn");
const initialBotMessage = "Hello, I am Med Predict assistant. Tell me your symptoms.";

function generateSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const chatSessionState = {
  sessionSymptoms: [],
  sessionId: generateSessionId()
};

function addMessage(text, type) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${type}`;
  bubble.textContent = text;
  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addMessageHtml(html, type) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${type}`;
  bubble.innerHTML = html;
  chatBox.appendChild(bubble);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatChatResponse(data) {
  const top3Text = (data.Top_3_Predictions || [])
    .map((item, idx) => `${idx + 1}) ${escapeHtml(item.Disease)} (${item.Probability_percent}%)`)
    .join(" | ");

  const precautions = (data.Recommended_Precautions || []).map((item) => escapeHtml(item)).join("; ");
  const followUps = (data.Follow_up_Questions || []).map((item) => escapeHtml(item)).join(" | ");
  const matchedSymptoms = (data.Matched_Symptoms || []).map((item) => escapeHtml(item)).join(", ");

  return `
    <div><strong>Possible Condition:</strong> ${escapeHtml(data.Possible_Condition || "Unknown")}</div>
    <div><strong>Confidence:</strong> ${Number(data.Confidence_percent || 0).toFixed(2)}%</div>
    <div><strong>Risk:</strong> ${escapeHtml(data.Risk_Level || "Unknown")} | <strong>Severity Score:</strong> ${escapeHtml(data.Severity_Score ?? "N/A")} | <strong>Emergency:</strong> ${data.Emergency ? "Yes" : "No"}</div>
    <div><strong>Matched Symptoms:</strong> ${matchedSymptoms || "None"}</div>
    <div><strong>Description:</strong> <strong>${escapeHtml(data.Description || "Not available")}</strong></div>
    <div><strong>Precautions:</strong> <strong>${precautions || "Not available"}</strong></div>
    ${top3Text ? `<div><strong>Top 3:</strong> ${top3Text}</div>` : ""}
    ${followUps ? `<div><strong>Follow-up:</strong> ${followUps}</div>` : ""}
    ${data.Disclaimer ? `<div><strong>Note:</strong> ${escapeHtml(data.Disclaimer)}</div>` : ""}
  `;
}

async function sendChatMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  chatInput.value = "";

  try {
    const data = await apiRequest(API_ENDPOINTS.chatbot, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: chatSessionState.sessionId,
        session_symptoms: chatSessionState.sessionSymptoms
      })
    });
    if (data.session_id) {
      chatSessionState.sessionId = data.session_id;
    }
    chatSessionState.sessionSymptoms = data.Session_Symptoms || [];
    if (data.Session_Reset) {
      chatBox.innerHTML = "";
      addMessage(initialBotMessage, "bot");
      addMessage(text, "user");
    }
    addMessageHtml(formatChatResponse(data), "bot");
  } catch (error) {
    addMessage(`Unable to process message: ${error.message}`, "bot");
  }
}

function startNewChat() {
  chatSessionState.sessionId = generateSessionId();
  chatSessionState.sessionSymptoms = [];
  chatBox.innerHTML = "";
  addMessage(initialBotMessage, "bot");
  chatInput.value = "";
  chatInput.focus();
}

const bookingForm = document.getElementById("bookingForm");
const bookingResult = document.getElementById("bookingResult");

async function handleBooking(event) {
  event.preventDefault();

  const payload = {
    full_name: document.getElementById("fullName").value.trim(),
    email: document.getElementById("email").value.trim(),
    date: document.getElementById("date").value,
    time: document.getElementById("time").value,
    department: document.getElementById("department").value,
    concern: document.getElementById("concern").value.trim()
  };

  if (Object.values(payload).some((value) => !value)) {
    bookingResult.classList.remove("hidden");
    bookingResult.innerHTML = `
      <h3>Form incomplete</h3>
      <p>Please fill all fields before booking.</p>
    `;
    return;
  }

  try {
    const data = await apiRequest(API_ENDPOINTS.booking, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    bookingResult.classList.remove("hidden");
    bookingResult.innerHTML = `
      <h3>${data.message}</h3>
      <p>${data.note}</p>
    `;
    bookingForm.reset();
  } catch (error) {
    bookingResult.classList.remove("hidden");
    bookingResult.innerHTML = `
      <h3>Booking failed</h3>
      <p>${error.message}</p>
    `;
  }
}

setupSmoothButtons();
loadSymptomsIntoDropdown();
renderSelectedSymptoms();

addSymptomBtn.addEventListener("click", addSelectedSymptom);
predictBtn.addEventListener("click", runPrediction);
clearSymptomsBtn.addEventListener("click", clearSymptoms);
skinFileInput.addEventListener("change", showSkinPreview);
sendChatBtn.addEventListener("click", sendChatMessage);
newChatBtn.addEventListener("click", startNewChat);
chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendChatMessage();
});
bookingForm.addEventListener("submit", handleBooking);
