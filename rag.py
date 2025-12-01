import os
import chromadb
from dotenv import load_dotenv 
from embeddings import embed_text 

load_dotenv() 

chroma_client = chromadb.PersistentClient( #initialiser un objet de chromaDB persistant
    path="/tmp/chromadb"  # le path de fichier qu'on va stocker la base vectorielle 
)
# Le paramètre 'path' pointe vers le répertoire où les données seront stockées.

collection = chroma_client.get_or_create_collection( #creation d'une collection ou import si est deja cree
    name="portfolio_rag",
    metadata={"hnsw:space": "cosine"} #utilisation de la fonction cos pour mesurer la similarite entre les vectures
)
# on utilise la fonction cosinus car c'est la meillere option dans NLP car il compare l'angle et ignore la magnitude

# La partie d'indexation des fichiers
def index_files():
    """ Indexe les fichiers de contenu du portfolio. """
    base_path = "./rag_content"
    
    if not os.path.exists(base_path): # si le ficher est n'existe pas
        print(f"ATTENTION : Le dossier {base_path} n'existe pas. Création...")
        os.makedirs(base_path)
        print("Veuillez y ajouter vos fichiers de contenu (ex: about.txt).")
        return

    files = os.listdir(base_path)
    if not files:
        print("Aucun fichier trouvé dans rag_content. L'index sera vide.")
        return
        
    print("Démarrage de l'indexation des documents")

    for filename in files:
        filepath = os.path.join(base_path, filename)
        
        if os.path.isdir(filepath) or not filename.endswith('.txt'):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        emb = embed_text(content) #transformer chaque texte en vecteur numerique

        collection.upsert( #pour l'ajout ou mis a jour le document dans chromaDB via son id 
            ids=[filename], #id unique de fichier 
            documents=[content], #contenu de fichier format texte
            embeddings=[emb] #contenu de fichier format vecteur 
        )
        print(f"Indexé : {filename}")

    print("Indexation terminée et sauvegardée")

# --- La partie de Recherche ---
def search(query: str, n_results=3) -> str: #cette fonction prend le texte de l'utilisateur et retourner le resultat pertinant
    """ 
    n_results et le nombre de document qu'on vas retourner on choisi 3 car on a une projet qui 
    utilise des petites fichiers et aussi reponses courtes 
    n_results est grand lorsque on a des fichiers grandes sinon 3 est le meillere pour les fichiers
    petites
    """
    query_emb = embed_text(query) #pour transformer en vecteur
    
    results = collection.query( 
        query_embeddings=[query_emb],
        n_results=n_results,
        include=['documents'] #pour recupere le texte des documentes n'est pas seulement les ids 
    )
    
    retrieved_docs = results['documents'][0] #recupere la liste des documents pour la 1ère requête
    return "\n---\n".join(retrieved_docs) #fusioner les documents par ---