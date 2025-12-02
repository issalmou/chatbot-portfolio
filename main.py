# main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import rag
from langdetect import detect
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://issalmouad.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

app = FastAPI(
    title="Issalmou Assistant AI ",
    version="1.0.0",
)

try:
    gemini_client = genai.Client()
except Exception:
    gemini_client = None


class ChatRequest(BaseModel):
    """Modèle pour la requête utilisateur."""

    query: str


# Cet événement s'exécute une seule fois au démarrage de l'API
@app.on_event("startup")
async def startup_event():
    """
    Indexe les fichiers du portfolio au démarrage de l'application
    si la base de données n'est pas déjà créée.
    """
    if gemini_client is None:
        print("ATTENTION: Clé GEMINI_API_KEY non chargée. La génération échouera.")

    rag.index_files()


# pour orienter gemini comment il va repondre
SYSTEM_INSTRUCTION = (
    "Tu es *Issalmou Assistant AI*, l’assistant virtuel officiel du portfolio d’Issalmou Adaaiche."
    "Tu ne dois jamais te présenter comme ChatGPT, Gemini, ou tout autre modèle d’IA."
    "Tu dois toujours te présenter comme l’assistant créé par Issalmou pour aider les visiteurs du portfolio."
    "Ta mission est de répondre de manière claire, professionnelle, concise et bienveillante."
    "Tu dois t'appuyer exclusivement sur les informations présentes dans le CONTEXTE fourni."
    "N’invente jamais de contenu. N’ajoute aucune information qui n’est pas explicitement présente dans le CONTEXTE."
    "Si l’utilisateur demande quelque chose qui ne figure pas dans le CONTEXTE :"
    "1. Réponds poliment que l’information n’est pas disponible, par exemple : "
    '"Désolé, cette information n\'est pas encore disponible dans le portfolio."'
    "2. Invite le visiteur à clarifier sa question ou à consulter la page 'Contact' ou à envoyer un email pour plus de détails."
    "Exemple de réponse complète : "
    "\"Désolé, cette information n'est pas encore disponible dans le portfolio. "
    "Si vous voulez, vous pouvez préciser votre question ou consulter la page 'Contact' pour plus de détails.\""
    "Si la question concerne une technologie ou une compétence d’Issalmou Adaaiche, explique-la de manière simple, professionnelle et adaptée au niveau du visiteur."
    "Tu peux reformuler, simplifier et améliorer la lisibilité de tes réponses tant que tu ne crées pas de nouvelles informations."
    "Ne commence pas par une salutation répétitive."
    "Réponds dans la même langue que la question de l’utilisateur, même si la langue par défaut est différente."
    "Important : Le nom du développeur 'Issalmou Adaaiche' doit rester inchangé."
)


# Fonction pour traduire un texte vers une langue cible
def translate_text(text: str, target_lang: str, isRag: bool) -> str:
    """
    Traduit un texte vers la langue cible sans modifier mon nom.
    Utilise un placeholder pour protéger le nom.
    """
    if len(text.strip()) < 4:
        return text
    placeholder = "<<ISSALMOU_ADAAICHE>>"
    text_protected = text.replace("Issalmou Adaaiche", placeholder)
    if isRag:
        prompt = f""" Tu vas traduire l'entrée utilisateur vers le français pour un moteur de recherche sémantique (RAG).
            Ne pas appliquer de règles linguistiques spécifiques.
            Ne pas détecter la Darija.
            Ne fais qu'une traduction simple et directe.
            Texte : "{text_protected}"
            """
    else:
        prompt = (
            "Évalue automatiquement la langue du message utilisateur.\n"
            "Si tu détectes que la langue est la Darija marocaine (arabe dialectal), alors :\n"
            f"- Ignore complètement {target_lang}\n"
            "- Réponds strictement dans la même langue que l'utilisateur (Darija).\n"
            "Sinon :\n"
            f"- Utilise {target_lang} pour traduire la réponse finale dans la langue demandée.\n"
            "\n"
            "Règles importantes :\n"
            f"- Traduire STRICTEMENT le texte ci-dessous vers {target_lang} si la langue n'est PAS la Darija.\n"
            "- Ne rien reformuler, ne rien retirer, ne rien ajouter.\n"
            "- Traduire uniquement les parties NON protégées.\n"
            "- Ne jamais modifier ou corriger le nom 'Issalmou Adaaiche'.\n"
            "- Ne jamais améliorer le style ou la qualité d'écriture.\n"
            "\n"
            f"Texte : {text_protected}\n"
            f"target_lang : {target_lang}"
        )

    result = gemini_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return result.text.replace(placeholder, "Issalmou Adaaiche")

@app.get("/chatbot")
async def chat():
    return {"response": "hello world i'am assistant AI of Issalmou "}

@app.post("/chatbot")
async def chat_endpoint(request: ChatRequest):
    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Le service Gemini n'est pas configuré (Clé API manquante).",
        )

    user_query = request.query

    detected_lang = detect(user_query)  # détection de la langue de la question "fr", "en", "ar", etc.

    lang_map = {
        "fr": "français",
        "en": "anglais",
        "ar": "arabe",
        "es": "espagnol",
        "af": "afrikaans",
        "bg": "bulgare",
        "bn": "bengali",
        "ca": "catalan",
        "cs": "tchèque",
        "da": "danois",
        "de": "allemand",
        "el": "grec",
        "et": "estonien",
        "fa": "persan",
        "fi": "finnois",
        "he": "hébreu",
        "hi": "hindi",
        "hr": "croate",
        "hu": "hongrois",
        "id": "indonésien",
        "it": "italien",
        "ja": "japonais",
        "ka": "géorgien",
        "ko": "coréen",
        "lt": "lituanien",
        "lv": "letton",
        "mk": "macédonien",
        "ms": "malais",
        "nb": "norvégien",
        "nl": "néerlandais",
        "pl": "polonais",
        "pt": "portugais",
        "ro": "roumain",
        "ru": "russe",
        "sk": "slovaque",
        "sl": "slovène",
        "sv": "suédois",
        "sw": "swahili",
        "ta": "tamoul",
        "th": "thaï",
        "tr": "turc",
        "uk": "ukrainien",
        "ur": "ourdou",
        "vi": "vietnamien",
        "zh": "chinois",
    }

    user_lang = lang_map.get(
        detected_lang, "français"
    )  # pour transformer le code en mot complete (exemple ar vers arabe) sinon on met francais

    # NE PAS traduire la question pour la RAG => embeddings en français
    question_for_rag = (
        user_query
        if user_lang == "français"
        else translate_text(user_query, "français",isRag=True)
    )

    # Récupération du contexte via RAG
    retrieved_context = rag.search(question_for_rag, n_results=3)

    # Construction du prompt final pour Gemini
    full_prompt = f"""
    {SYSTEM_INSTRUCTION}
    --- CONTEXTE RÉCUPÉRÉ (Pour la réponse seulement) ---
    {retrieved_context}
    --- FIN CONTEXTE ---

    QUESTION DE L'UTILISATEUR : {user_query}
    Réponds uniquement à la question, ne te présente pas et ne répète aucune salutation.
    Réponds dans la même langue que la question de l’utilisateur ({user_lang}), même si la langue par défaut est différente.
    Important : Le nom du développeur 'Issalmou Adaaiche' doit rester inchangé.
    """

    try:
        # generation d'une réponse via gemini en utilisons le contexte a partir chromaDB et la question d'utilisateur
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",  # C’est un modèle de génération rapide, puissant et capable de produire du texte humainement cohérent.
            contents=full_prompt,
        )
        # Si la langue de l’utilisateur n’est pas français, traduire la réponse en conséquence
        final_response = (
            response.text
            if user_lang == "français"
            else translate_text(response.text, user_lang,isRag=False)
        )

        return {"response": final_response}

    except Exception as e:
        print(f"Erreur de génération Gemini: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la génération de la réponse.",
        )
