from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyngrok import ngrok
import numpy as np
import json

# Flask app
app = Flask(__name__)

# Embed dosyaları
embeddings = np.load("prompts_embeddings.npy")
with open("prompts_texts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
with open("responses_texts.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

# (RAM yetiyorsa)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@app.route("/sor", methods=["POST"])
def cevap_ver():
    veri = request.json
    soru = veri.get("mesaj", "")
    embed = model.encode([soru])
    skorlar = cosine_similarity(embed, embeddings)[0]
    en_yakin_index = int(np.argmax(skorlar))
    return jsonify({"cevap": responses[en_yakin_index]})

# Ngrok aç ve Flask başlat
if __name__ == "__main__":
    public_url = ngrok.connect(8080)
    print(f"🔗 Ngrok Linkin: {public_url}")
    app.run(debug=True, port=8080)