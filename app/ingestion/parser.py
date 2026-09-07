"""Parsing SÛR (sans exécution de code) du fichier source de vérité du
portfolio : `translations.js` (objet JS exporté par défaut, consommé par le
frontend React/i18n).

Le fichier est un littéral d'objet JavaScript statique (chaînes, nombres,
booléens, null, tableaux/objets imbriqués — pas de fonction ni d'expression
calculée), avec clés non quotées et virgules finales : exactement le
sous-ensemble couvert par JSON5. On utilise donc `json5` (parseur basé sur
une grammaire, jamais eval/exec/Node.js) pour le transformer en dict Python ;
toute expression non littérale ferait échouer le parsing proprement
(SyntaxError) plutôt que d'être exécutée.

Une partie du contenu (SEO, libellés d'UI) n'a aucune valeur informative pour
le RAG et polluerait le retrieval si elle était indexée : ce module ne retient
donc que les sections narratives/factuelles réelles (voir
`_extract_content_blocks`), en réutilisant les libellés déjà traduits comme
préfixes de contexte plutôt que d'en inventer des fixes non traduits.
"""

from __future__ import annotations

from dataclasses import dataclass

import json5

_LANG_ALIASES = {
    "fr": "fr", "français": "fr", "francais": "fr", "french": "fr",
    "en": "en", "english": "en", "anglais": "en",
    "ar": "ar", "arabe": "ar", "arabic": "ar", "العربية": "ar",
}


class PortfolioParseError(Exception):
    """Le fichier source ne peut pas être interprété comme un translations.js valide."""


@dataclass(frozen=True)
class ContentBlock:
    lang: str
    section: str       # clé machine stable, ex. "resume", "projects", "contact"
    subsection: str    # clé machine stable, ex. "education.0", "agep", "info"
    label: str         # préfixe de contexte lisible, déjà traduit (extrait du fichier)
    text: str
    topic_group: str   # identité language-agnostique = "section:subsection"
    entity_type: str   # catégorie d'entité listable, ex. "project", "education"
    entity_id: str      # identifiant stable de l'entité au sein de son type


def derive_entity(section: str, subsection: str) -> tuple[str, str]:
    """Dérive (entity_type, entity_id) à partir des clés machine stables du
    parser : "projects" -> ("project", shortName) ; une subsection à point
    (ex. "education.0") -> (préfixe, subsection entier) ; sinon (sous-section
    singleton) -> (subsection, subsection). Contrairement à `topic_group`
    (dédup cross-langue d'un même sujet), ceci identifie l'entité réelle au
    sein d'une catégorie à plusieurs instances (ex. 5 projets) — nécessaire
    pour qu'une question exhaustive retrouve bien chaque projet."""
    if section == "projects":
        return "project", subsection
    if "." in subsection:
        prefix, _, _ = subsection.partition(".")
        return prefix, subsection
    return subsection, subsection


def _extract_object_literal(js_source: str) -> str:
    """Isole le littéral d'objet `{ ... }` assigné à `translations`, en
    ignorant les accolades qui pourraient apparaître à l'intérieur de
    chaînes de caractères (scan caractère par caractère, sensible aux
    guillemets et aux échappements)."""
    try:
        start = js_source.index("{")
    except ValueError as exc:
        raise PortfolioParseError("Aucun littéral d'objet trouvé dans le fichier.") from exc

    depth = 0
    in_string = False
    quote = ""
    escape = False
    for i in range(start, len(js_source)):
        c = js_source[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                in_string = False
        else:
            if c in ('"', "'", "`"):
                in_string = True
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return js_source[start : i + 1]

    raise PortfolioParseError("Accolade fermante correspondante introuvable (fichier tronqué ou invalide).")


def _clean_list(items) -> list[str]:
    return [str(i) for i in (items or []) if i is not None and str(i).strip()]


def _join_defined(*parts: str | None, sep: str = " — ") -> str:
    """Joint uniquement les parties réellement présentes, pour ne jamais
    laisser un littéral "None" s'infiltrer dans le texte indexé quand un
    champ optionnel de translations.js est absent."""
    return sep.join(str(p) for p in parts if p)


def _extract_content_blocks(lang: str, data: dict) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []

    def add(section: str, subsection: str, label: str, parts: list[str]) -> None:
        text = "\n\n".join(p.strip() for p in parts if p and str(p).strip())
        if text:
            entity_type, entity_id = derive_entity(section, subsection)
            blocks.append(
                ContentBlock(
                    lang=lang, section=section, subsection=subsection,
                    label=label, text=text, topic_group=f"{section}:{subsection}",
                    entity_type=entity_type, entity_id=entity_id,
                )
            )

    home = data.get("home") or {}
    about = data.get("about") or {}
    resume = data.get("resume") or {}
    services = data.get("services") or {}
    projects = data.get("projects") or {}
    project_details_labels = data.get("projectDetails") or {}
    contact = data.get("contact") or {}

    # metaDescription est le seul endroit où le nom complet "Issalmou Adaaiche"
    # apparaît (le reste du contenu narratif n'utilise que le prénom) ; le
    # reste de l'arbre `seo` reste exclu, ce n'est pas du langage naturel exploitable.
    add(
        "about", "intro", about.get("title", "About"),
        [data.get("metaDescription"), home.get("heroDescription"), about.get("headline"), about.get("description"), about.get("quote")],
    )

    # --- Expertise (accueil) ---
    home_services = home.get("services") or []
    if home_services:
        lines = [_join_defined(s.get("title"), s.get("description"), sep=": ") for s in home_services if s.get("title")]
        add("home", "expertise", home.get("servicesTitle", "Expertise"), ["\n".join(lines)])

    # --- Formation ---
    for i, edu in enumerate(resume.get("education") or []):
        degree = _join_defined(edu.get("degree"), f"({edu['period']})" if edu.get("period") else None, sep=" ")
        line = _join_defined(degree or None, edu.get("institution"))
        add("resume", f"education.{i}", resume.get("educationTitle", "Education"), [line])

    # --- Compétences (pourcentages) ---
    skills_data = resume.get("skillsData") or []
    if skills_data:
        lines = [f"{s.get('name')}: {s.get('value')}%" for s in skills_data if s.get("name")]
        add("resume", "skills_data", resume.get("skillsTitle", "Skills"), ["\n".join(lines)])

    # --- Stages / expériences professionnelles ---
    for i, intern in enumerate(resume.get("internships") or []):
        company_duration = _join_defined(intern.get("company"), f"({intern['duration']})" if intern.get("duration") else None, sep=" ")
        header = _join_defined(intern.get("role"), company_duration or None)
        responsibilities = "\n".join(f"- {r}" for r in _clean_list(intern.get("responsibilities")))
        add("resume", f"internship.{i}", resume.get("internshipsTitle", "Experience"), [header, responsibilities])

    # --- Services proposés ---
    for i, item in enumerate(services.get("items") or []):
        header = _join_defined(item.get("title"), item.get("desc"), sep=": ")
        details = "; ".join(_clean_list(item.get("list")))
        add("services", f"service.{i}", services.get("title", "Services"), [header, details])

    # --- Processus de travail ---
    process = services.get("process") or {}
    steps = process.get("steps") or []
    if steps:
        lines = [_join_defined(s.get("title"), s.get("desc"), sep=": ") for s in steps if s.get("title")]
        add("services", "process", f"{services.get('title', 'Services')} > {process.get('title', 'Process')}", ["\n".join(lines)])

    # --- Projets ---
    for project in (projects.get("projects_details") or []):
        short_name = project.get("shortName")
        if not short_name:
            continue  # identifiant stable indisponible : impossible de garantir un chunk_id fiable
        title = project.get("title")
        header = f"{title}" + (f" ({project['company']})" if project.get("company") else "")
        date_type = ", ".join(v for v in (project.get("date"), project.get("type")) if v)
        if date_type:
            header += f" — {date_type}"
        challenge_label = project_details_labels.get("theChallenge", "Challenge")
        solution_label = project_details_labels.get("theSolution", "Solution")
        features_label = project_details_labels.get("keyFeatures", "Key features")
        parts = [
            header,
            project.get("description"),
            project.get("projectOverview"),
            f"{challenge_label}: {project.get('theChallenge')}" if project.get("theChallenge") else None,
            f"{solution_label}: {project.get('theSolution')}" if project.get("theSolution") else None,
            f"{features_label}: " + ", ".join(_clean_list(project.get("keyFeatures"))) if project.get("keyFeatures") else None,
            "Technologies: " + ", ".join(_clean_list(project.get("technologies"))) if project.get("technologies") else None,
            project.get("externalUrl"),
        ]
        add("projects", short_name, f"{projects.get('title', 'Projects')} > {title}", [p for p in parts if p])

    # --- Contact ---
    contact_lines = [contact.get("sectionDescription"), contact.get("contactInfoDescription")]
    info_bits = [v for v in (contact.get("location"), contact.get("phone"), contact.get("email")) if v]
    if info_bits:
        contact_lines.append(" — ".join(info_bits))
    add("contact", "info", contact.get("sectionTitle", "Contact"), contact_lines)

    return blocks


def parse_translations_js(raw_text: str) -> list[ContentBlock]:
    if not raw_text or not raw_text.strip():
        raise PortfolioParseError("Le fichier est vide.")

    object_literal = _extract_object_literal(raw_text)

    try:
        data = json5.loads(object_literal)
    except Exception as exc:
        raise PortfolioParseError(f"Impossible d'analyser le contenu comme un objet JS de données ({exc}).") from exc

    if not isinstance(data, dict):
        raise PortfolioParseError("La structure de premier niveau doit être un objet.")

    blocks: list[ContentBlock] = []
    for raw_key, value in data.items():
        lang = _LANG_ALIASES.get(str(raw_key).lower())
        if lang is None or not isinstance(value, dict):
            continue  # clé de premier niveau non reconnue comme une langue : ignorée, pas une erreur
        blocks.extend(_extract_content_blocks(lang, value))

    if not blocks:
        raise PortfolioParseError(
            "Aucune section de contenu exploitable reconnue. Le fichier doit contenir au moins "
            "un bloc de langue valide (en/fr/ar) avec du contenu dans les sections attendues."
        )

    return blocks
