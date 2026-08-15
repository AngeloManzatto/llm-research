"""
Created on Sat Aug 15 12:43:03 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

from src.tasks.sft.conversation.dataset_compiler.build.executor import (
    execute_relation_build,
    execute_scenario_build,
)

from src.tasks.sft.conversation.dataset_compiler.build.registry import (
    RELATION_BUILD_SPECS,
    SCENARIO_BUILD_SPECS,
)

###############################################################################
# Build Stage 0
###############################################################################

def build_stage0() -> list[dict]:
    """
    Build all currently registered Stage 0 relation-based dataset units.

    Each registered build specification is executed independently and
    the resulting rows are concatenated into one candidate dataset.
    """

    rows: list[dict] = []

    for spec in RELATION_BUILD_SPECS:
        rows.extend(
            execute_relation_build(spec)
        )

    for spec in SCENARIO_BUILD_SPECS:
        rows.extend(
            execute_scenario_build(spec)
        )

    return rows

###############################################################################
# Build Report Stage 0
###############################################################################

def build_stage0_with_report() -> tuple[list[dict], dict[str, int]]:
    """
    Build all registered Stage 0 relation-based units and return
    both the generated rows and the number of rows per build unit.
    """

    rows: list[dict] = []
    report: dict[str, int] = {}

    for spec in RELATION_BUILD_SPECS:
        build_rows = execute_relation_build(spec)

        rows.extend(build_rows)
        report[spec.id] = len(build_rows)

    for spec in SCENARIO_BUILD_SPECS:
        build_rows = execute_scenario_build(spec)

        rows.extend(build_rows)
        report[spec.id] = len(build_rows)

    return rows, report

