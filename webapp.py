from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import os

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), "model")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load label encoder
label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), "label_encoder.pkl"))

# Telugu and English advice
advice_dict = {
    "Psoriasis": {
        "en": "Use medicated creams and moisturizers. Consult a dermatologist.",
        "te": "ఔషధ క్రీమ్‌లు మరియు మాయిశ్చరైజర్‌లు వాడండి. చర్మ వైద్యుడిని సంప్రదించండి."
    },
    "Varicose Veins": {
        "en": "Avoid standing too long. Use compression stockings.",
        "te": "ఎక్కువసేపు నిలబడటం నివారించండి. కంప్రెషన్ స్టాకింగ్‌లు వాడండి."
    },
    "Typhoid": {
        "en": "Take antibiotics as prescribed. Maintain hygiene and rest.",
        "te": "డాక్టర్ చెప్పిన విధంగా యాంటిబయోటిక్స్ తీసుకోండి. పరిశుభ్రత పాటించండి, విశ్రాంతి తీసుకోండి."
    },
    "Chicken pox": {
        "en": "Avoid scratching. Take antiviral medicines and rest.",
        "te": "గోకడం నివారించండి. యాంటీవైరల్ మందులు తీసుకోండి, విశ్రాంతి అవసరం."
    },
    "Impetigo": {
        "en": "Use topical antibiotics. Avoid skin contact with others.",
        "te": "టాపికల్ యాంటిబయోటిక్స్ వాడండి. ఇతరులతో చర్మ స్పర్శ నివారించండి."
    },
    "Dengue": {
        "en": "Drink plenty of fluids. Monitor platelet count and rest well.",
        "te": "ఎక్కువగా ద్రవ పదార్థాలు తీసుకోండి. ప్లేట్లెట్ కౌంట్‌ను గమనించండి మరియు విశ్రాంతి తీసుకోండి."
    },
    "Fungal infection": {
        "en": "Use antifungal creams. Keep the area clean and dry.",
        "te": "యాంటీఫంగల్ క్రీమ్‌లు వాడండి. శుభ్రంగా, పొడిగా ఉంచండి."
    },
    "Common Cold": {
        "en": "Stay hydrated and take rest. Use decongestants if needed.",
        "te": "ద్రవాలు ఎక్కువగా తీసుకోండి, విశ్రాంతి తీసుకోండి. అవసరమైతే డీకాన్జెస్టెంట్లు వాడండి."
    },
    "Pneumonia": {
        "en": "Take prescribed antibiotics. Seek immediate medical help if severe.",
        "te": "డాక్టర్ సూచించిన యాంటిబయోటిక్స్ తీసుకోండి. తీవ్రతగా ఉంటే తక్షణ వైద్య సహాయం తీసుకోండి."
    },
    "Dimorphic Hemorrhoids": {
        "en": "Eat high-fiber food. Avoid straining during bowel movements.",
        "te": "ఫైబర్ ఎక్కువగా ఉన్న ఆహారం తీసుకోండి. మలవిసర్జన సమయంలో ఒత్తిడి వద్దు."
    },
    "Arthritis": {
        "en": "Exercise gently. Take anti-inflammatory medicines.",
        "te": "సాధ్యమైనంత వరకు వ్యాయామం చేయండి. శోథ నిరోధక మందులు వాడండి."
    },
    "Acne": {
        "en": "Clean your skin regularly. Avoid oily foods.",
        "te": "చర్మాన్ని పరిశుభ్రంగా ఉంచండి. నూనె అధికంగా ఉన్న ఆహారం నివారించండి."
    },
    "Bronchial Asthma": {
        "en": "Avoid allergens and pollution. Use inhalers as directed.",
        "te": "అలర్జీ మరియు కాలుష్యం నుండి దూరంగా ఉండండి. డాక్టర్ చెప్పిన విధంగా ఇన్హేలర్ వాడండి."
    },
    "Hypertension": {
        "en": "Reduce salt intake. Monitor blood pressure regularly.",
        "te": "ఉప్పు తగ్గించండి. రక్తపోటును క్రమం తప్పకుండా గమనించండి."
    },
    "Migraine": {
        "en": "Avoid triggers like noise and light. Take prescribed medicines.",
        "te": "శబ్దం మరియు వెలుతురు వంటి కారణాలు నివారించండి. సూచించిన మందులు వాడండి."
    },
    "Cervical spondylosis": {
        "en": "Do neck exercises. Avoid long screen time.",
        "te": "మెడ వ్యాయామాలు చేయండి. ఎక్కువసేపు స్క్రీన్‌కి దూరంగా ఉండండి."
    },
    "Jaundice": {
        "en": "Avoid oily food. Take rest and stay hydrated.",
        "te": "నూనె ఎక్కువగా ఉన్న ఆహారం తినవద్దు. విశ్రాంతి తీసుకోండి, ద్రవ పదార్థాలు ఎక్కువగా తీసుకోండి."
    },
    "Malaria": {
        "en": "Take antimalarial medicines. Prevent mosquito bites.",
        "te": "యాంటీమలేరియా మందులు తీసుకోండి. దోమ కాటు నివారించండి."
    },
    "urinary tract infection": {
        "en": "Drink more water. Consult doctor for antibiotics.",
        "te": "ఎక్కువగా నీరు త్రాగండి. యాంటిబయోటిక్స్ కోసం డాక్టర్‌ను సంప్రదించండి."
    },
    "allergy": {
        "en": "Avoid allergens. Take antihistamines.",
        "te": "అలర్జీ కారకాలను నివారించండి. యాంటీహిస్టమిన్లు తీసుకోండి."
    },
    "gastroesophageal reflux disease": {
        "en": "Avoid spicy foods. Don’t lie down right after eating.",
        "te": "కారం ఎక్కువగా ఉన్న ఆహారం తినవద్దు. భోజనం తర్వాత వెంటనే పడుకోకండి."
    },
    "drug reaction": {
        "en": "Stop the suspected medication. Seek medical help immediately.",
        "te": "అనుమానిత మందును ఆపండి. తక్షణ వైద్య సహాయం తీసుకోండి."
    },
    "peptic ulcer disease": {
        "en": "Avoid spicy and acidic foods. Take antacids as advised.",
        "te": "కారం మరియు ఆమ్లాహారాన్ని నివారించండి. సూచించిన ఆంటాసిడ్స్ తీసుకోండి."
    },
    "diabetes": {
        "en": "Control blood sugar levels. Maintain a healthy diet.",
        "te": "చక్కెర స్థాయిని నియంత్రించండి. తగిన ఆహారం తీసుకోండి."
    }
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    lang = data.get("lang", "en")  # default to English

    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**encoding)
    pred_class = torch.argmax(outputs.logits, dim=1).item()

    disease = label_encoder.inverse_transform([pred_class])[0]
    advice = advice_dict.get(disease, {}).get(lang, "Please consult a doctor.")

    return jsonify({"disease": disease, "advice": advice})

if __name__ == "__main__":
    app.run(debug=True)

