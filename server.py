from flask import Flask, request, jsonify
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = np.load("prompts_embeddings.npy")

with open("prompts_texts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
with open("responses_texts.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

def chatbot_sor(soru):
    embed = model.encode([soru])
    skorlar = cosine_similarity(embed, embeddings)[0]
    en_yakin_index = int(np.argmax(skorlar))
    return responses[en_yakin_index]

@app.route("/sor", methods=["POST"])
def cevap_ver():
    veri = request.json
    soru = veri.get("mesaj", "")
    yanit = chatbot_sor(soru)
    return jsonify({"cevap": yanit})

if __name__ == "__main__":
    app.run(debug=True)