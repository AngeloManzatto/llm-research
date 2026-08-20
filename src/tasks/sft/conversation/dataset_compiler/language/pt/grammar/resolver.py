"""
Created on Sun Aug 16 22:12:06 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Build Grammar Index
###############################################################################

def build_grammar_index(
    forms: list[dict],
) -> dict:
    """
    Build a lookup index from loaded grammar-form rows.
    """

    index = {}

    for item in forms:
        form = item["form"]
        features = item["features"]
        value = item["value"]

        feature_key = tuple(
            sorted(features.items())
        )

        key = (
            form,
            feature_key,
        )

        if key in index:
            raise ValueError(
                f"Duplicate grammar form for "
                f"{form!r} with features {features!r}."
            )

        index[key] = value

    return index

###############################################################################
# Resolve Form
###############################################################################

def resolve_form(
    grammar_index: dict,
    *,
    form: str,
    features: dict[str, str],
) -> str:
    """
    Resolve one grammatical surface form.

    Example:

        form="article.definite"
        features={
            "Gender": "Fem",
            "Number": "Plur",
        }

        -> "as"
    """

    feature_key = tuple(
        sorted(features.items())
    )

    key = (
        form,
        feature_key,
    )

    try:
        return grammar_index[key]

    except KeyError as exc:
        raise KeyError(
            f"No grammar realization found for "
            f"form={form!r}, features={features!r}."
        ) from exc