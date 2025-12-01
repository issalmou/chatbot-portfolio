# embeddings.py
from google import genai
from dotenv import load_dotenv

load_dotenv()

try:
    client = genai.Client() # pour instancier un client pour le but d'utiliser ce objet dans la transformation d'une phrase vers une vecteur numerique
except Exception as e:
    print("Erreur Client Gemini :", e)
    client = None


def embed_text(text: str) -> list: #cette fonction recovoir une texte et retourner le texte en vecteur 
    if client is None:
        raise ConnectionError("Client Gemini non configuré.")

    result = client.models.embed_content(
        model="gemini-embedding-001",   # car 'gemini-embedding-001' est ployvalent aussi supporté plusieurs langues ce type est général
        contents=[text]              
    )

    return list(result.embeddings[0].values) #  retourne une vecteur des valeurs represent la phrase en vecteur 
