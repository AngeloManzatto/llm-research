# -*- coding: utf-8 -*-
"""
Collects every relation into one list. Adding a new relation means:
create relations/your_relation.py defining a Relation instance, import
it here, add it to RELATIONS. Nothing else needs to change — same
principle as the metrics/ package's registry.
"""

from .animal_sound import animal_sound
from .animal_baby import animal_baby
from .shape_sides import shape_sides
from .color_of import color_of
from .profession import profession
from .animal_group import animal_group
from .opposite_of import opposite_of

RELATIONS = [
    animal_sound,
    animal_baby,
    shape_sides,
    color_of,
    profession,
    animal_group,
    opposite_of,
]
