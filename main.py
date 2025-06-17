from flask import Flask, jsonify, request
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load Dataset
df = pd.read_excel("/workspaces/cyber-ai/cybercrime_dummy_data_10k.xlsx")
df['Hour'] = pd.to_datetime(df['Time of Incident'], format='%H:%M:%S').dt.hour

# Load Models
tfidf = joblib.load("/workspaces/cyber-ai/tfidf_vectorizer.joblib")
text_classifier = joblib.load("/workspaces/cyber-ai/text_classifier_model.joblib")

# Predefined Mitigation Steps
attack_guidelines = {
    "Ransomware": [
        "Disconnect from the internet and isolate the infected system.",
        "Do not pay the ransom.",
        "Restore data from verified backups if available.",
        "Report the incident to cyber authorities."
    ],
    "Phishing": [
        "Do not click on any links or attachments.",
        "Reset affected passwords immediately.",
        "Enable two-factor authentication.",
        "Report the email or message to the relevant team."
    ],
    "Scam": [
        "Do not share personal or financial information.",
        "Block and report the scammer.",
        "Monitor your bank account for suspicious activity."
    ],
    "DDoS": [
        "Contact your hosting provider for support.",
        "Deploy DDoS mitigation tools or services.",
        "Monitor traffic using firewalls and network logs."
    ],
    "Malware": [
        "Run a full antivirus/malware scan.",
        "Disconnect infected devices from the network.",
        "Update and patch your software regularly."
    ],
    "Hacking": [
        "Reset passwords and check for unauthorized access.",
        "Audit user accounts and privileges.",
        "Enable security logging and monitor activity."
    ]
}

@app.route('/api/heatmap')
def heatmap():
    data = df.groupby('Hour')['Incident ID'].count().reset_index(name='Count')
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/top-keywords')
def top_keywords():
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    X = vectorizer.fit_transform(df['Attack Description'].astype(str))
    keywords = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    return jsonify([{ "keyword": kw, "count": int(ct) } for kw, ct in zip(keywords, counts)])

@app.route('/api/platform-distribution')
def platform_distribution():
    result = df['Platform'].value_counts().reset_index()
    result.columns = ['Platform', 'Count']
    return jsonify(result.to_dict(orient='records'))

@app.route('/api/attack-types')
def attack_types():
    result = df['Attack Type'].value_counts().reset_index()
    result.columns = ['Attack Type', 'Count']
    return jsonify(result.to_dict(orient='records'))

@app.route('/api/flagged-links')
def flagged_links():
    links = df[df['Suspicious Link'] != '']['Suspicious Link'].value_counts().reset_index()
    links.columns = ['Suspicious Link', 'Count']
    return jsonify(links.to_dict(orient='records'))

@app.route('/api/pattern-analysis')
def pattern_analysis():
    pattern = df.groupby(['Device Used', 'Attack Type'])['Incident ID'].count().reset_index()
    pattern.columns = ['Device Used', 'Attack Type', 'Count']
    return jsonify(pattern.to_dict(orient='records'))

@app.route('/api/predict-attack-type', methods=['POST'])
def predict_attack_type():
    try:
        data = request.get_json(force=True)
        desc = data.get("description", "").strip()
        if not desc:
            return jsonify({"error": "Description is required."}), 400

        X_input = tfidf.transform([desc])
        prediction = text_classifier.predict(X_input)[0]
        confidence = max(text_classifier.predict_proba(X_input)[0])
        mitigation = attack_guidelines.get(prediction, ["Please consult a cybersecurity expert."])

        return jsonify({
            "predicted_attack_type": prediction,
            "confidence": round(confidence, 2),
            "mitigation_steps": mitigation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "✅ Cybercrime AI Pattern Detection API is running."

if __name__ == "__main__":
    app.run(debug=True)