"""
Created on Tue Jul 28 22:44:04 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from __future__ import annotations

from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.core.graph import FactGraph
from src.tasks.sft.conversation.dataset_compiler.core.models import Fact, Node

###############################################################################
# Test data
###############################################################################


def make_dog() -> Node:
    return Node(
        id="animal.dog",
        node_type="animal",
        labels={
            "en": "dog",
            "pt": "cachorro",
        },
    )


def make_cat() -> Node:
    return Node(
        id="animal.cat",
        node_type="animal",
        labels={
            "en": "cat",
            "pt": "gato",
        },
    )


def make_puppy() -> Node:
    return Node(
        id="animal.puppy",
        node_type="animal_young",
        labels={
            "en": "puppy",
            "pt": "filhote de cachorro",
        },
    )


def make_kitten() -> Node:
    return Node(
        id="animal.kitten",
        node_type="animal_young",
        labels={
            "en": "kitten",
            "pt": "filhote de gato",
        },
    )


def make_bark() -> Node:
    return Node(
        id="sound.bark",
        node_type="animal_sound",
        labels={
            "en": "bark",
            "pt": "latido",
        },
    )


def make_meow() -> Node:
    return Node(
        id="sound.meow",
        node_type="animal_sound",
        labels={
            "en": "meow",
            "pt": "miado",
        },
    )


def make_graph() -> FactGraph:
    """
    Create a fresh graph for each test.

    Tests should not share graph state.
    """
    graph = FactGraph()

    graph.add_nodes(
        [
            make_dog(),
            make_cat(),
            make_puppy(),
            make_kitten(),
            make_bark(),
            make_meow(),
        ]
    )

    return graph


###############################################################################
# Small assertion helper
###############################################################################


def assert_raises(
    expected_exception: type[BaseException],
    function: Callable[[], object],
    *,
    message_contains: str | None = None,
) -> BaseException:
    """
    Assert that calling a function raises the expected exception.

    This replaces pytest.raises for direct execution inside Spyder.
    """
    try:
        function()
    except expected_exception as exc:
        if (
            message_contains is not None
            and message_contains not in str(exc)
        ):
            raise AssertionError(
                f"Expected exception message to contain "
                f"{message_contains!r}, but received: {str(exc)!r}"
            ) from exc

        return exc
    except Exception as exc:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, "
            f"but {type(exc).__name__} was raised instead: {exc}"
        ) from exc

    raise AssertionError(
        f"Expected {expected_exception.__name__}, "
        "but no exception was raised."
    )


###############################################################################
# Node tests
###############################################################################


def test_add_node() -> None:
    graph = FactGraph()
    dog = make_dog()

    graph.add_node(dog)

    assert graph.has_node("animal.dog")
    assert graph.get_node("animal.dog") == dog
    assert graph.node_count == 1


def test_readding_same_node_is_idempotent() -> None:
    graph = FactGraph()
    dog = make_dog()

    graph.add_node(dog)
    graph.add_node(dog)

    assert graph.node_count == 1


def test_conflicting_node_is_rejected() -> None:
    graph = FactGraph()
    graph.add_node(make_dog())

    conflicting_node = Node(
        id="animal.dog",
        node_type="vehicle",
        labels={
            "en": "car",
            "pt": "carro",
        },
    )

    assert_raises(
        ValueError,
        lambda: graph.add_node(conflicting_node),
        message_contains="already registered with different data",
    )


def test_unknown_node_lookup_is_rejected() -> None:
    graph = FactGraph()

    assert_raises(
        KeyError,
        lambda: graph.get_node("animal.unknown"),
        message_contains="is not registered",
    )


def test_iter_nodes_filters_by_type() -> None:
    graph = make_graph()

    animal_ids = {
        node.id
        for node in graph.iter_nodes(node_type="animal")
    }

    assert animal_ids == {
        "animal.dog",
        "animal.cat",
    }


###############################################################################
# Fact tests
###############################################################################


def test_add_fact() -> None:
    graph = make_graph()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    graph.add_fact(fact)

    assert graph.has_fact(
        "animal.dog",
        "animal_baby",
        "animal.puppy",
    )

    assert graph.get_fact(
        "animal.dog",
        "animal_baby",
        "animal.puppy",
    ) == fact

    assert graph.fact_count == 1


def test_readding_same_fact_is_idempotent() -> None:
    graph = make_graph()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    graph.add_fact(fact)
    graph.add_fact(fact)

    assert graph.fact_count == 1


def test_fact_with_unknown_subject_is_rejected() -> None:
    graph = make_graph()

    fact = Fact(
        subject_id="animal.unknown",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    assert_raises(
        KeyError,
        lambda: graph.add_fact(fact),
        message_contains="subject node",
    )


def test_fact_with_unknown_object_is_rejected() -> None:
    graph = make_graph()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="animal.unknown",
    )

    assert_raises(
        KeyError,
        lambda: graph.add_fact(fact),
        message_contains="object node",
    )


def test_unknown_fact_lookup_is_rejected() -> None:
    graph = make_graph()

    assert_raises(
        KeyError,
        lambda: graph.get_fact(
            "animal.dog",
            "animal_baby",
            "animal.puppy",
        ),
        message_contains="is not registered",
    )


def test_node_can_participate_in_multiple_relations() -> None:
    graph = make_graph()

    graph.add_facts(
        [
            Fact(
                subject_id="animal.dog",
                relation_id="animal_baby",
                object_id="animal.puppy",
            ),
            Fact(
                subject_id="animal.dog",
                relation_id="animal_sound",
                object_id="sound.bark",
            ),
        ]
    )

    baby_nodes = graph.objects(
        subject_id="animal.dog",
        relation_id="animal_baby",
    )

    sound_nodes = graph.objects(
        subject_id="animal.dog",
        relation_id="animal_sound",
    )

    assert [node.id for node in baby_nodes] == [
        "animal.puppy",
    ]

    assert [node.id for node in sound_nodes] == [
        "sound.bark",
    ]


def test_iter_facts_filters_by_relation() -> None:
    graph = make_graph()

    graph.add_facts(
        [
            Fact(
                subject_id="animal.dog",
                relation_id="animal_baby",
                object_id="animal.puppy",
            ),
            Fact(
                subject_id="animal.dog",
                relation_id="animal_sound",
                object_id="sound.bark",
            ),
        ]
    )

    facts = list(
        graph.iter_facts(
            relation_id="animal_baby",
        )
    )

    assert len(facts) == 1
    assert facts[0].subject_id == "animal.dog"
    assert facts[0].object_id == "animal.puppy"


def test_iter_facts_combines_filters() -> None:
    graph = make_graph()

    graph.add_facts(
        [
            Fact(
                subject_id="animal.dog",
                relation_id="animal_baby",
                object_id="animal.puppy",
            ),
            Fact(
                subject_id="animal.cat",
                relation_id="animal_baby",
                object_id="animal.kitten",
            ),
        ]
    )

    facts = list(
        graph.iter_facts(
            subject_id="animal.cat",
            relation_id="animal_baby",
        )
    )

    assert len(facts) == 1
    assert facts[0].object_id == "animal.kitten"


def test_subjects_returns_connected_subject_nodes() -> None:
    graph = make_graph()

    graph.add_fact(
        Fact(
            subject_id="animal.dog",
            relation_id="animal_baby",
            object_id="animal.puppy",
        )
    )

    subjects = graph.subjects(
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    assert [node.id for node in subjects] == [
        "animal.dog",
    ]


def test_relation_ids() -> None:
    graph = make_graph()

    graph.add_facts(
        [
            Fact(
                subject_id="animal.dog",
                relation_id="animal_baby",
                object_id="animal.puppy",
            ),
            Fact(
                subject_id="animal.dog",
                relation_id="animal_sound",
                object_id="sound.bark",
            ),
        ]
    )

    assert graph.relation_ids() == {
        "animal_baby",
        "animal_sound",
    }


def test_graph_length_is_fact_count() -> None:
    graph = make_graph()

    graph.add_fact(
        Fact(
            subject_id="animal.dog",
            relation_id="animal_baby",
            object_id="animal.puppy",
        )
    )

    assert len(graph) == 1


def test_node_membership() -> None:
    graph = make_graph()

    assert "animal.dog" in graph
    assert "animal.unknown" not in graph


###############################################################################
# Spyder test runner
###############################################################################


TESTS: tuple[Callable[[], None], ...] = (
    test_add_node,
    test_readding_same_node_is_idempotent,
    test_conflicting_node_is_rejected,
    test_unknown_node_lookup_is_rejected,
    test_iter_nodes_filters_by_type,
    test_add_fact,
    test_readding_same_fact_is_idempotent,
    test_fact_with_unknown_subject_is_rejected,
    test_fact_with_unknown_object_is_rejected,
    test_unknown_fact_lookup_is_rejected,
    test_node_can_participate_in_multiple_relations,
    test_iter_facts_filters_by_relation,
    test_iter_facts_combines_filters,
    test_subjects_returns_connected_subject_nodes,
    test_relation_ids,
    test_graph_length_is_fact_count,
    test_node_membership,
)


def run_tests() -> None:
    """
    Execute all tests and print a compact report.
    """
    passed = 0
    failed = 0

    print("=" * 79)
    print("FactGraph tests")
    print("=" * 79)

    for test_function in TESTS:
        test_name = test_function.__name__

        try:
            test_function()
        except Exception as exc:
            failed += 1
            print(
                f"[FAILED] {test_name}\n"
                f"         {type(exc).__name__}: {exc}"
            )
        else:
            passed += 1
            print(f"[PASSED] {test_name}")

    print("-" * 79)
    print(
        f"Results: {passed} passed, "
        f"{failed} failed, "
        f"{len(TESTS)} total"
    )
    print("=" * 79)

    if failed:
        raise AssertionError(
            f"{failed} graph test(s) failed."
        )


if __name__ == "__main__":
    run_tests()