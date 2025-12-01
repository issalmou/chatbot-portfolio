# main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import rag
from langdetect import detect

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
    "Tu peux reformuler, simplifier et améliorer la lisibilité de tes réponses tant que tu ne crées pas de nouvelles informations."
    "Ne commence pas par une salutation répétitive."
)


# Fonction pour traduire un texte vers une langue cible
def translate_text(text: str, target_lang: str) -> str:
    """
    Traduit un texte vers la langue cible sans modifier mon nom.
    Utilise un placeholder pour protéger le nom.
    """
    placeholder = "<<ISSALMOU_ADAAICHE>>"
    text_protected = text.replace("Issalmou Adaaiche", placeholder)

    prompt = f"Traduire le texte suivant en {target_lang} sans rien modifier sauf traduire le reste :\n{text_protected}"
    result = gemini_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return result.text.replace(placeholder, "Issalmou Adaaiche")


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
        "af": "afrikaans",
        "ar": "arabe",
        "bg": "bulgare",
        "bn": "bengali",
        "ca": "catalan",
        "cs": "tchèque",
        "da": "danois",
        "de": "allemand",
        "el": "grec",
        "en": "anglais",
        "es": "espagnol",
        "et": "estonien",
        "fa": "persan",
        "fi": "finnois",
        "fr": "français",
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
    
    user_lang = lang_map.get(detected_lang, "français") # pour transformer le code en mot complete (exemple ar vers arabe) sinon on met francais

    # Traduire la question vers le français pour le RAG
    if user_lang != "français":
        translated_query = translate_text(user_query, "français")
    else:
        translated_query = user_query

    # Récupération du contexte via RAG
    retrieved_context = rag.search(translated_query, n_results=3)
    
    # Construction du prompt final pour Gemini
    full_prompt = f"""
    {SYSTEM_INSTRUCTION}

    --- CONTEXTE RÉCUPÉRÉ (Pour la réponse seulement) ---
    {retrieved_context}
    --- FIN CONTEXTE ---

    QUESTION DE L'UTILISATEUR : {user_query}
    
    Réponds uniquement à la question, dans la même langue que la question de l’utilisateur,
    Réponds uniquement à la question, ne te présente pas et ne répète aucune salutation.
    Important : Le nom du développeur 'Issalmou Adaaiche' doit rester inchangé.
    """

    try:
        # generation d'une réponse via gemini en utilisons le contexte a partir chromaDB et la question d'utilisateur
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",  # C’est un modèle de génération rapide, puissant et capable de produire du texte humainement cohérent.
            contents=full_prompt,
        )
        return {"response": response.text}

    except Exception as e:
        print(f"Erreur de génération Gemini: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la génération de la réponse.",
        )
