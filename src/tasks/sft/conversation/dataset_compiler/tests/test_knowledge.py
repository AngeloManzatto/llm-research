"""
Created on Sun Aug  9 14:02:36 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.core.graph import FactGraph
from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
    Node,
    RelationDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.validation.knowledge import (
    validate_fact_types,
)

###############################################################################
# Test Helpers
###############################################################################


def make_graph() -> FactGraph:
    graph = FactGraph()

    graph.add_nodes(
        [
            Node(
                id="animal.dog",
                node_type="animal",
                labels={
                    "en": "dog",
                    "pt": "cachorro",
                },
            ),
            Node(
                id="animal.puppy",
                node_type="animal_young",
                labels={
                    "en": "puppy",
                    "pt": "filhote de cachorro",
                },
            ),
            Node(
                id="sound.bark",
                node_type="animal_sound",
                labels={
                    "en": "bark",
                    "pt": "latido",
                },
            ),
        ]
    )

    return graph


def make_animal_baby_relation() -> RelationDefinition:
    return RelationDefinition(
        id="animal_baby",
        subject_type="animal",
        object_type="animal_young",
    )


def assert_raises(
    expected_exception: type[BaseException],
    function: Callable[[], object],
    *,
    message_contains: str | None = None,
) -> None:
    try:
        function()

    except expected_exception as exc:
        if (
            message_contains is not None
            and message_contains not in str(exc)
        ):
            raise AssertionError(
                f"Expected message containing "
                f"{message_contains!r}, got {str(exc)!r}."
            ) from exc

        return

    except Exception as exc:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, "
            f"but got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"Expected {expected_exception.__name__}, "
        "but no exception was raised."
    )


###############################################################################
# Tests
###############################################################################


def test_valid_fact_passes() -> None:
    graph = make_graph()
    relation = make_animal_baby_relation()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    validate_fact_types(
        graph=graph,
        fact=fact,
        relation=relation,
    )


def test_wrong_relation_is_rejected() -> None:
    graph = make_graph()
    relation = make_animal_baby_relation()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_sound",
        object_id="animal.puppy",
    )

    assert_raises(
        ValueError,
        lambda: validate_fact_types(
            graph=graph,
            fact=fact,
            relation=relation,
        ),
        message_contains="does not match",
    )


def test_wrong_subject_type_is_rejected() -> None:
    graph = make_graph()
    relation = make_animal_baby_relation()

    fact = Fact(
        subject_id="sound.bark",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    assert_raises(
        ValueError,
        lambda: validate_fact_types(
            graph=graph,
            fact=fact,
            relation=relation,
        ),
        message_contains="requires subject type",
    )


def test_wrong_object_type_is_rejected() -> None:
    graph = make_graph()
    relation = make_animal_baby_relation()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="sound.bark",
    )

    assert_raises(
        ValueError,
        lambda: validate_fact_types(
            graph=graph,
            fact=fact,
            relation=relation,
        ),
        message_contains="requires object type",
    )


def test_unknown_subject_is_rejected() -> None:
    graph = make_graph()
    relation = make_animal_baby_relation()

    fact = Fact(
        subject_id="animal.unknown",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    assert_raises(
        KeyError,
        lambda: validate_fact_types(
            graph=graph,
            fact=fact,
            relation=relation,
        ),
    )


###############################################################################
# Runner
###############################################################################


TESTS = (
    test_valid_fact_passes,
    test_wrong_relation_is_rejected,
    test_wrong_subject_type_is_rejected,
    test_wrong_object_type_is_rejected,
    test_unknown_subject_is_rejected,
)


def run_tests() -> None:
    passed = 0
    failed = 0

    print("=" * 79)
    print("Knowledge validation tests")
    print("=" * 79)

    for test in TESTS:
        try:
            test()

        except Exception as exc:
            failed += 1
            print(
                f"[FAILED] {test.__name__}\n"
                f"         {type(exc).__name__}: {exc}"
            )

        else:
            passed += 1
            print(f"[PASSED] {test.__name__}")

    print("-" * 79)
    print(
        f"Results: {passed} passed, "
        f"{failed} failed, "
        f"{len(TESTS)} total"
    )
    print("=" * 79)

    if failed:
        raise AssertionError(
            f"{failed} knowledge validation test(s) failed."
        )


if __name__ == "__main__":
    run_tests()