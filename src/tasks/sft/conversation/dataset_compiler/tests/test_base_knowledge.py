"""
Created on Sun Aug  9 14:40:21 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from collections.abc import Callable

from src.tasks.sft.conversation.dataset_compiler.core.models import (
    Fact,
    Node,
    RelationDefinition,
)
from src.tasks.sft.conversation.dataset_compiler.knowledge.base import (
    KnowledgeBase,
)


###############################################################################
# Test Data
###############################################################################


def make_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()

    kb.add_nodes(
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

    kb.add_relation(
        RelationDefinition(
            id="animal_baby",
            subject_type="animal",
            object_type="animal_young",
        )
    )

    return kb


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


def test_valid_fact_is_inserted() -> None:
    kb = make_knowledge_base()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    kb.add_fact(fact)

    assert kb.fact_count == 1

    assert kb.get_fact(
        "animal.dog",
        "animal_baby",
        "animal.puppy",
    ) == fact


def test_unknown_relation_is_rejected() -> None:
    kb = make_knowledge_base()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_sound",
        object_id="sound.bark",
    )

    assert_raises(
        KeyError,
        lambda: kb.add_fact(fact),
        message_contains="is not registered",
    )

    assert kb.fact_count == 0


def test_invalid_subject_type_is_rejected() -> None:
    kb = make_knowledge_base()

    fact = Fact(
        subject_id="sound.bark",
        relation_id="animal_baby",
        object_id="animal.puppy",
    )

    assert_raises(
        ValueError,
        lambda: kb.add_fact(fact),
        message_contains="requires subject type",
    )

    assert kb.fact_count == 0


def test_invalid_object_type_is_rejected() -> None:
    kb = make_knowledge_base()

    fact = Fact(
        subject_id="animal.dog",
        relation_id="animal_baby",
        object_id="sound.bark",
    )

    assert_raises(
        ValueError,
        lambda: kb.add_fact(fact),
        message_contains="requires object type",
    )

    assert kb.fact_count == 0


def test_readding_same_relation_is_idempotent() -> None:
    kb = KnowledgeBase()

    relation = RelationDefinition(
        id="animal_baby",
        subject_type="animal",
        object_type="animal_young",
    )

    kb.add_relation(relation)
    kb.add_relation(relation)

    assert kb.relation_count == 1


def test_conflicting_relation_is_rejected() -> None:
    kb = KnowledgeBase()

    kb.add_relation(
        RelationDefinition(
            id="animal_baby",
            subject_type="animal",
            object_type="animal_young",
        )
    )

    conflicting_relation = RelationDefinition(
        id="animal_baby",
        subject_type="animal",
        object_type="animal_sound",
    )

    assert_raises(
        ValueError,
        lambda: kb.add_relation(conflicting_relation),
        message_contains="already registered with different data",
    )

    assert kb.relation_count == 1


###############################################################################
# Runner
###############################################################################


TESTS = (
    test_valid_fact_is_inserted,
    test_unknown_relation_is_rejected,
    test_invalid_subject_type_is_rejected,
    test_invalid_object_type_is_rejected,
    test_readding_same_relation_is_idempotent,
    test_conflicting_relation_is_rejected,
)


def run_tests() -> None:
    passed = 0
    failed = 0

    print("=" * 79)
    print("KnowledgeBase tests")
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
            f"{failed} KnowledgeBase test(s) failed."
        )


if __name__ == "__main__":
    run_tests()