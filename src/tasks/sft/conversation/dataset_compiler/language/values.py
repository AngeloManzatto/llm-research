"""
Created on Mon Aug 17 07:01:17 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

###############################################################################
# Grammar Values
###############################################################################

def grammar_values(
    *,
    prefix: str,
    grammar_index: dict,
    value_specs: dict[str, str],
    features: dict[str, str],
) -> dict[str, str]:
    """
    Resolve multiple grammatical forms into namespaced render values.

    Parameters
    ----------
    prefix:
        Namespace used for the output keys.

        Example:
            "subject"

    grammar_index:
        Indexed grammatical realizations produced by
        build_grammar_index().

    value_specs:
        Mapping between output-key suffixes and grammatical form IDs.

        Example:
            {
                "def_article": "article.definite",
                "de_def_article": "article.de_definite",
            }

    features:
        Grammatical features used to resolve each form.

        Example:
            {
                "Gender": "Fem",
                "Number": "Sing",
            }

    Returns
    -------
    dict[str, str]

        Example:
            {
                "subject_def_article": "a",
                "subject_de_def_article": "da",
            }
    """

    if not prefix.strip():
        raise ValueError(
            "Grammar-value prefix cannot be empty."
        )

    if not value_specs:
        return {}

    if not features:
        raise ValueError(
            "Grammar features cannot be empty."
        )

    from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.resolver import (
        resolve_form,
    )

    values: dict[str, str] = {}

    for key_suffix, form_id in value_specs.items():

        output_key = f"{prefix}_{key_suffix}"

        values[output_key] = resolve_form(
            grammar_index,
            form=form_id,
            features=features,
        )

    return values