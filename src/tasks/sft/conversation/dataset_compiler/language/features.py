"""
Created on Mon Aug 17 06:41:52 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

###############################################################################
# Resolve Node Features
###############################################################################

def resolve_node_features(
    *,
    node,
    feature_spec: dict,
) -> dict[str, str]:
    """
    Resolve grammatical features for one node.

    feature_spec example:

        {
            "metadata": {
                "Gender": "pt_gender",
            },
            "defaults": {
                "Number": "Sing",
            },
        }
    """

    features: dict[str, str] = {}

    for feature_name, metadata_key in (
        feature_spec.get("metadata", {}).items()
    ):
        try:
            features[feature_name] = node.metadata[
                metadata_key
            ]
        except KeyError as exc:
            raise ValueError(
                f"Node {node.id!r} does not define metadata "
                f"{metadata_key!r} required for grammatical "
                f"feature {feature_name!r}."
            ) from exc

    features.update(
        feature_spec.get("defaults", {})
    )

    return features