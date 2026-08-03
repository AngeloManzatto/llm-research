from dataset_compiler.core.models import Fact


ANIMAL_FACTS = (
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
    Fact(
        subject_id="animal.dog",
        relation_id="animal_sound",
        object_id="sound.bark",
    ),
    Fact(
        subject_id="animal.cat",
        relation_id="animal_sound",
        object_id="sound.meow",
    ),
)