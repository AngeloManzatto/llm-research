from dataset_compiler.core.graph import FactGraph

from .entities import ANIMAL_NODES
from .facts import ANIMAL_FACTS


def build_animal_graph() -> FactGraph:
    graph = FactGraph()
    graph.add_nodes(ANIMAL_NODES)
    graph.add_facts(ANIMAL_FACTS)
    return graph