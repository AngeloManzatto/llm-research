# -*- coding: utf-8 -*-
"""
base.py — the core of the relation-table generator.

Follows the pattern confirmed across real QA-dataset literature
(LC-QuAD, GraphQuestions, KQA Pro, INFOSEEK, ReasonVQA, KGQuest):
facts live as structured (subject, answer) pairs under a named
relation — not free text — and each relation has SEVERAL question
templates, not one. Generation is the cross-join of every fact against
every template for that relation and language.

This is what directly fixes the duplication problem we spent this
whole project chasing by hand: a relation with 40 facts and 4 templates
produces 160 genuinely distinct rows, deterministically, with zero
risk of the "obvious answer" convergence that free-text LLM generation
kept producing (Whiskers, Rex, blue, Jupiter, opposite-of-hot).

Multiple templates per relation is not an afterthought bolted on later
-- it's required at definition time, confirmed as standard practice
(ReasonVQA: "To boost linguistic diversity, we also develop linguistic
variations for each template").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Relation:
    """
    One relation = one fact table + several question templates, per
    language. `facts` maps subject -> answer. `templates` are strings
    containing a single {subject} placeholder.

    name       : short identifier, used in generated row IDs.
    category   : which Stage 0 category these rows belong to.
    facts      : {"en": {subject: answer, ...}, "pt": {...}}
    templates  : {"en": [template, ...], "pt": [template, ...]}
    """
    name: str
    category: str
    facts: dict[str, dict[str, str]]
    templates: dict[str, list[str]]

    def __post_init__(self):
        for lang in self.facts:
            if lang not in self.templates or not self.templates[lang]:
                raise ValueError(f"Relation {self.name!r}: no templates defined for language {lang!r}")
        for lang, tpls in self.templates.items():
            for t in tpls:
                if "{subject}" not in t:
                    raise ValueError(f"Relation {self.name!r}: template missing {{subject}} placeholder: {t!r}")

    def n_facts(self, language: str) -> int:
        return len(self.facts.get(language, {}))

    def n_templates(self, language: str) -> int:
        return len(self.templates.get(language, []))

    def max_rows(self, language: str) -> int:
        """Full cross-join size for one language — the ceiling on how
        many distinct rows this relation can produce without repeating
        a (subject, template) pair."""
        return self.n_facts(language) * self.n_templates(language)


def generate_rows(relation: Relation, language: str, id_prefix: str | None = None) -> list[dict]:
    """
    Cross-joins every fact against every template for one language,
    producing schema-valid 2-message training rows. Exhaustive by
    default — every (subject, template) pair is used exactly once, so
    output size is deterministic and there is no sampling-induced
    convergence risk to check for afterward.
    """
    facts = relation.facts.get(language, {})
    templates = relation.templates.get(language, [])
    prefix = id_prefix or relation.name

    rows = []
    idx = 0
    for subject, answer in facts.items():
        for template in templates:
            idx += 1
            question = template.format(subject=subject)
            question = question[0].upper() + question[1:] if question else question
            rows.append({
                "id": f"{relation.category}_{language}_{prefix}_{idx:05d}",
                "category": relation.category,
                "language": language,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            })
    return rows
