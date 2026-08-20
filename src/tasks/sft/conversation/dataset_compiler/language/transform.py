"""
Created on Wed Aug 19 08:03:00 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from pathlib import Path

from src.tasks.sft.conversation.dataset_compiler.compiler.transform import (
    Transforms,
)

from src.tasks.sft.conversation.dataset_compiler.language.features import (
    resolve_node_features,
)

from src.tasks.sft.conversation.dataset_compiler.language.io import (
    load_json,
    load_jsonl,
    load_value_specs,
)

from src.tasks.sft.conversation.dataset_compiler.language.pt.grammar.resolver import (
    build_grammar_index,
)

from src.tasks.sft.conversation.dataset_compiler.language.values import (
    grammar_values,
)


###############################################################################
# Create Grammar Transform
###############################################################################

def create_grammar_transform(
    *,
    language: str,
    node_role: str,
    grammar_dir: str | Path,
) -> Transforms:
    """
    Create a grammar transform for one language and semantic node role.

    Grammar data is loaded once when the transform is created.
    """

    grammar_dir = Path(grammar_dir)

    feature_spec = load_json(
        grammar_dir / "feature_spec.json"
    )

    forms = load_jsonl(
        grammar_dir / "forms.jsonl"
    )

    value_specs = load_value_specs(
        grammar_dir / "value_specs.jsonl"
    )

    grammar_index = build_grammar_index(
        forms
    )

    def transform(
        knowledge_base,
        fact,
        template,
    ) -> dict[str, str]:

        if template.language != language:
            return {}
        
        node_id_attribute = f"{node_role}_id"
        
        try:
            node_id = getattr(
                fact,
                node_id_attribute,
            )
        except AttributeError as exc:
            raise ValueError(
                f"Source {type(fact).__name__!r} does not define "
                f"node role {node_role!r}."
            ) from exc

        node = knowledge_base.get_node(
            node_id
        )

        features = resolve_node_features(
            node=node,
            feature_spec=feature_spec,
        )

        return grammar_values(
            prefix=node_role,
            grammar_index=grammar_index,
            value_specs=value_specs,
            features=features,
        )

    return transform