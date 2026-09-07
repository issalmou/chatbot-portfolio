"""Fixtures et doublures de test.

Tous les tests automatisés tournent SANS réseau : ni Chroma Cloud ni les
providers LLM (Gemini/Mistral/Groq/OpenAI) ne sont appelés réellement.
`FakeChromaStore` réimplémente l'interface publique de `ChromaStore` en
mémoire pure (aucun SDK Chroma), ce qui démontre au passage l'abstraction
voulue : le RAG (app/rag/retrieval.py) ne sait pas si le store est local,
distant ou un simple dict Python.

La vérification contre les vraies API (modèles réellement disponibles,
qualité de retrieval, fallback réel) a été faite manuellement en direct avec
de vraies clés — voir le rapport final pour le détail et les résultats. Un
test d'intégration réel (marqué et ignoré si les credentials manquent) est
fourni séparément dans test_pipeline.py pour Chroma Cloud.
"""

from __future__ import annotations

import pytest

from app.llm.base import LLMMessage, LLMProvider, LLMProviderError, LLMResult

_KEYWORDS = ["agep", "esticar", "contact", "profil", "profile", "frontend", "backend", "issalmou", "resume", "education"]


def fake_vector(text: str) -> list[float]:
    """Embedding factice mais déterministe : un vecteur one-hot par mot-clé
    détecté. Suffisant pour tester le MÉCANISME de retrieval (cross-langue,
    déduplication par sujet, cache) sans dépendre d'un vrai modèle sémantique."""
    lowered = text.lower()
    vector = [1.0 if kw in lowered else 0.0 for kw in _KEYWORDS]
    vector.append(0.01)  # évite le vecteur nul
    return vector


class FakeEmbedder:
    """Double de EmbeddingProvider (app/embeddings/base.py)."""

    model_id = "fake-e5"
    model_version = "test"

    def __init__(self) -> None:
        self.call_count = 0
        self.document_call_count = 0
        self.query_call_count = 0

    @property
    def dimension(self) -> int:
        return len(_KEYWORDS) + 1

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.document_call_count += 1
        return [fake_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.call_count += 1
        self.query_call_count += 1
        return fake_vector(text)


class FakeLLMProvider(LLMProvider):
    def __init__(self, name: str, configured: bool = True, error: LLMProviderError | None = None, reply: str = "ok"):
        self.name = name
        self.model = "fake-model"  # exposé publiquement, comme les vrais providers (voir manager.current_provider())
        self._configured = configured
        self._error = error
        self._reply = reply
        self.call_count = 0

    def is_configured(self) -> bool:
        return self._configured

    def generate(self, messages: list[LLMMessage]) -> LLMResult:
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return LLMResult(text=self._reply, provider=self.name, model=self.model, latency_ms=1.0)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeChromaStore:
    """Double en mémoire de ChromaStore — même contrat public, aucun réseau."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    @property
    def is_available(self) -> bool:
        return True

    def get_meta(self) -> dict:
        record = self._data.get("__source_meta__")
        return dict(record["metadata"]) if record else {}

    def get_content_version(self) -> str | None:
        return self.get_meta().get("source_hash")

    def commit_new_version(self, source_hash: str, chunk_count: int) -> None:
        self._data["__source_meta__"] = {
            "document": "source_meta",
            "embedding": [1.0],
            "metadata": {"kind": "meta", "source_hash": source_hash, "chunk_count": chunk_count},
        }

    def existing_chunk_hashes(self) -> dict[str, str]:
        return {
            doc_id: rec["metadata"]["chunk_hash"]
            for doc_id, rec in self._data.items()
            if rec["metadata"].get("kind") == "chunk" and "chunk_hash" in rec["metadata"]
        }

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        return {i: self._data[i]["embedding"] for i in ids if i in self._data}

    def sync(self, to_upsert: list[dict], ids_to_delete: list[str]) -> None:
        for u in to_upsert:
            self._data[u["id"]] = {
                "document": u["document"],
                "embedding": u["embedding"],
                "metadata": {**u["metadata"], "kind": "chunk"},
            }
        for doc_id in ids_to_delete:
            self._data.pop(doc_id, None)

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        candidates = [(doc_id, rec) for doc_id, rec in self._data.items() if rec["metadata"].get("kind") == "chunk"]
        scored = sorted(
            ((1 - _cosine(embedding, rec["embedding"]), doc_id, rec) for doc_id, rec in candidates),
            key=lambda t: t[0],
        )
        deduped: list[dict] = []
        seen_topics: set = set()
        for dist, _doc_id, rec in scored:
            topic = rec["metadata"].get("topic_group")
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            deduped.append({"document": rec["document"], "metadata": rec["metadata"], "distance": dist})
            if len(deduped) >= top_k:
                break
        return deduped

    def count(self) -> int:
        return sum(1 for rec in self._data.values() if rec["metadata"].get("kind") == "chunk")

    def get_all_by_entity_type(self, entity_type: str, preferred_lang: str | None = None) -> tuple[list[dict], int]:
        """Miroir de ChromaStore.get_all_by_entity_type (filtrage metadata,
        dédup par entity_id avec préférence de langue, parties recollées)."""
        candidates = [
            rec for rec in self._data.values()
            if rec["metadata"].get("kind") == "chunk" and rec["metadata"].get("entity_type") == entity_type
        ]
        by_entity: dict[str, dict[str, dict[int, tuple[str, dict]]]] = {}
        for rec in candidates:
            meta = rec["metadata"]
            entity_id = meta.get("entity_id")
            if not entity_id:
                continue
            lang = meta.get("lang", "")
            part_index = meta.get("part_index", 0)
            by_entity.setdefault(entity_id, {}).setdefault(lang, {})[part_index] = (rec["document"], meta)

        lang_priority = [preferred_lang] if preferred_lang else []
        lang_priority += [l for l in ("en", "fr", "ar") if l != preferred_lang]

        merged: list[dict] = []
        for entity_id, by_lang in by_entity.items():
            chosen_lang = next((l for l in lang_priority if l in by_lang), next(iter(by_lang)))
            parts = by_lang[chosen_lang]
            ordered_docs = [parts[idx][0] for idx in sorted(parts)]
            representative_meta = parts[min(parts)][1]
            merged.append({"document": "\n".join(ordered_docs), "metadata": representative_meta, "entity_id": entity_id})
        return merged, len(candidates)

    def get_by_entity_id(self, entity_id: str, preferred_lang: str | None = None) -> dict | None:
        """Miroir de ChromaStore.get_by_entity_id."""
        candidates = [
            rec for rec in self._data.values()
            if rec["metadata"].get("kind") == "chunk" and rec["metadata"].get("entity_id") == entity_id
        ]
        if not candidates:
            return None
        by_lang: dict[str, dict[int, tuple[str, dict]]] = {}
        for rec in candidates:
            meta = rec["metadata"]
            lang = meta.get("lang", "")
            part_index = meta.get("part_index", 0)
            by_lang.setdefault(lang, {})[part_index] = (rec["document"], meta)
        lang_priority = [preferred_lang] if preferred_lang else []
        lang_priority += [l for l in ("en", "fr", "ar") if l != preferred_lang]
        chosen_lang = next((l for l in lang_priority if l in by_lang), next(iter(by_lang)))
        parts = by_lang[chosen_lang]
        ordered_docs = [parts[idx][0] for idx in sorted(parts)]
        representative_meta = parts[min(parts)][1]
        return {"document": "\n".join(ordered_docs), "metadata": representative_meta, "entity_id": entity_id}


@pytest.fixture()
def tmp_store() -> FakeChromaStore:
    return FakeChromaStore()


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --- Fixture translations.js minimale, structurellement fidèle au vrai fichier ---
# (mêmes clés réelles : home.heroDescription, about.*, resume.education/skillsData/
# internships, services.items/process, projects.projects_details[].shortName,
# projectDetails.*, contact.*, plus du "bruit" volontaire — nav/seo/chatbot —
# pour vérifier qu'il est bien exclu du RAG.)
SAMPLE_TRANSLATIONS_JS = r"""
const translations = {
    en: {
        metaDescription: "Portfolio bio noise-free sentence.",
        seo: { home: { title: "seomarker", description: "seomarker" } },
        nav: { home: "Home" },
        heroName: "Issalmou Adaaiche",
        home: {
            heroDescription: "I am Issalmou Adaaiche, a Full-Stack developer.",
            servicesTitle: "Expertise",
            services: [ { title: "Backend", description: "Node.js and APIs." } ],
        },
        about: {
            title: "About",
            headline: "Curious developer",
            description: "I build things.",
            quote: "\"Technology is my tool.\"",
        },
        resume: {
            title: "Resume",
            educationTitle: "Education",
            education: [
                { degree: "BSc Computer Science", period: "2020-2023", institution: "University X" }
            ],
            skillsTitle: "Skills",
            skillsData: [ { name: "Backend", value: 95 } ],
            internshipsTitle: "Experience",
            internships: [
                { role: "Dev Intern", duration: "3 months", company: "ACME",
                  responsibilities: ["Built REST APIs.", "Fixed bugs."] }
            ],
        },
        services: {
            title: "Services",
            items: [ { title: "Web Dev", desc: "Custom websites.", list: ["Responsive", "Fast"] } ],
            process: { title: "Process", steps: [ { title: "Discovery", desc: "Understand needs." } ] },
        },
        projects: {
            title: "Projects",
            projects_details: [
                {
                    title: "AGEP", shortName: "agep", company: "Clinic", date: "2024", type: "Web",
                    description: "AGEP manages paramedical teams.",
                    projectOverview: "Overview of AGEP.",
                    theChallenge: "Coordinate schedules.",
                    theSolution: "A web platform.",
                    keyFeatures: ["Automation", "Dashboards"],
                    technologies: ["Laravel", "MySQL"],
                    externalUrl: null,
                    images: ["/img/agep.png"],
                    seo: { title: "noise", description: "noise" }
                },
            ],
        },
        projectDetails: { theChallenge: "The Challenge", theSolution: "The Solution", keyFeatures: "Key Features" },
        contact: {
            sectionTitle: "Contact",
            sectionDescription: "Reach out.",
            contactInfoDescription: "Always open to talk.",
            location: "Laayoune, Morocco",
            phone: "+212 640065118",
            email: "issalmouadaaiche@gmail.com",
            formFields: { name: "Your Name" },
        },
        notFound: { title: "Not found" },
        chatbot: { placeholder: "Type your message...", assistantError: "Sorry." },
    },
    fr: {
        metaDescription: "Phrase bio du portfolio sans bruit.",
        seo: { home: { title: "seomarker" } },
        nav: { home: "Accueil" },
        heroName: "Issalmou Adaaiche",
        home: {
            heroDescription: "Je suis Issalmou Adaaiche, développeur Full-Stack.",
            servicesTitle: "Expertise",
            services: [ { title: "Backend", description: "Node.js et API." } ],
        },
        about: {
            title: "À propos",
            headline: "Développeur curieux",
            description: "Je construis des choses.",
            quote: "\"La technologie est mon outil.\"",
        },
        resume: {
            title: "CV",
            educationTitle: "Formation",
            education: [
                { degree: "Licence Informatique", period: "2020-2023", institution: "Université X" }
            ],
            skillsTitle: "Compétences",
            skillsData: [ { name: "Backend", value: 95 } ],
            internshipsTitle: "Expérience",
            internships: [
                { role: "Stagiaire Dev", duration: "3 mois", company: "ACME",
                  responsibilities: ["A développé des API REST.", "A corrigé des bugs."] }
            ],
        },
        services: {
            title: "Services",
            items: [ { title: "Dev Web", desc: "Sites sur mesure.", list: ["Responsive", "Rapide"] } ],
            process: { title: "Processus", steps: [ { title: "Découverte", desc: "Comprendre les besoins." } ] },
        },
        projects: {
            title: "Projets",
            projects_details: [
                {
                    title: "AGEP", shortName: "agep", company: "Clinique", date: "2024", type: "Web",
                    description: "AGEP gère les équipes paramédicales.",
                    projectOverview: "Aperçu d'AGEP.",
                    theChallenge: "Coordonner les plannings.",
                    theSolution: "Une plateforme web.",
                    keyFeatures: ["Automatisation", "Tableaux de bord"],
                    technologies: ["Laravel", "MySQL"],
                    externalUrl: null,
                },
            ],
        },
        projectDetails: { theChallenge: "Le défi", theSolution: "La solution", keyFeatures: "Fonctionnalités clés" },
        contact: {
            sectionTitle: "Contact",
            sectionDescription: "Contactez-moi.",
            contactInfoDescription: "Toujours ouvert à la discussion.",
            location: "Laâyoune, Maroc",
            phone: "+212 640065118",
            email: "issalmouadaaiche@gmail.com",
            formFields: { name: "Votre Nom" },
        },
        notFound: { title: "Introuvable" },
        chatbot: { placeholder: "Écris ton message...", assistantError: "Désolé." },
    },
    ar: {
        heroName: "إسلمو إيدعيش",
        home: {
            heroDescription: "أنا Issalmou Adaaiche، مطور ويب متكامل.",
            servicesTitle: "الخبرات",
            services: [ { title: "الواجهة الخلفية", description: "Node.js وواجهات برمجية." } ],
        },
        about: {
            title: "حول",
            headline: "مطور فضولي",
            description: "أبني أشياء.",
            quote: "\"التكنولوجيا هي أداتي.\"",
        },
        resume: {
            title: "السيرة الذاتية",
            educationTitle: "التعليم",
            education: [ { degree: "إجازة في الإعلاميات", period: "2020-2023", institution: "جامعة X" } ],
            skillsTitle: "المهارات",
            skillsData: [ { name: "الواجهة الخلفية", value: 95 } ],
            internshipsTitle: "الخبرة",
            internships: [
                { role: "متدرب", duration: "3 أشهر", company: "ACME", responsibilities: ["طوّر واجهات برمجية.", "أصلح الأخطاء."] }
            ],
        },
        services: {
            title: "الخدمات",
            items: [ { title: "تطوير الويب", desc: "مواقع مخصصة.", list: ["متجاوب"] } ],
            process: { title: "العملية", steps: [ { title: "الاكتشاف", desc: "فهم الاحتياجات." } ] },
        },
        projects: {
            title: "المشاريع",
            projects_details: [
                {
                    title: "AGEP", shortName: "agep", company: "عيادة", date: "2024", type: "ويب",
                    description: "AGEP يدير الفرق شبه الطبية.",
                    projectOverview: "نظرة عامة على AGEP.",
                    theChallenge: "تنسيق الجداول.",
                    theSolution: "منصة ويب.",
                    keyFeatures: ["أتمتة"],
                    technologies: ["Laravel", "MySQL"],
                    externalUrl: null,
                },
            ],
        },
        projectDetails: { theChallenge: "التحدي", theSolution: "الحل", keyFeatures: "الميزات الرئيسية" },
        contact: {
            sectionTitle: "اتصل بي",
            sectionDescription: "تواصل معي.",
            contactInfoDescription: "منفتح دائمًا للنقاش.",
            location: "العيون، المغرب",
            phone: "+212 640065118",
            email: "issalmouadaaiche@gmail.com",
        },
        chatbot: { placeholder: "اكتب رسالتك..." },
    },
};
export default translations;
"""
