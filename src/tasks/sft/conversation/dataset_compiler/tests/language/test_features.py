"""
Created on Mon Aug 17 06:47:14 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.core.models import Node

from src.tasks.sft.conversation.dataset_compiler.language.features import (
    resolve_node_features,
)

###############################################################################
# Node
###############################################################################

node = Node(
    id="object.banana",
    node_type="object",
    labels={
        "en": "banana",
        "pt": "banana",
    },
    metadata={
        "pt_gender": "Fem",
    },
)

###############################################################################
# Feature Specification
###############################################################################

feature_spec = {
    "metadata": {
        "Gender": "pt_gender",
    },
    "defaults": {
        "Number": "Sing",
    },
}

###############################################################################
# Node Features
###############################################################################

features = resolve_node_features(
    node=node,
    feature_spec=feature_spec,
)

print(features)

assert features == {
    "Gender": "Fem",
    "Number": "Sing",
}

print("Node feature resolution OK.")