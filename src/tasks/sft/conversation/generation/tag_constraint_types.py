"""
Created on Sat Aug  8 11:06:57 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
"""
tag_constraint_types.py — semi-automated migration helper for
instruction_following rows.

Infers constraint_type (and constraint_value where needed) from the
instruction phrasing itself, using pattern matching against the known
constraint types constraint_satisfied.py actually supports. Anything
that doesn't confidently match a known pattern is left UNTAGGED and
flagged for manual review, rather than guessed at silently -- a wrong
guess here is worse than no guess, since it would make the metric
confidently score the wrong thing instead of just not running yet.

This does NOT touch your files directly -- it reads rows, reports what
it would tag and what it couldn't, so you can review before applying
anything. Adjust the patterns below to match YOUR actual instruction
phrasings; these are reasonable guesses at common template wording, not
guaranteed to match what you actually generated.
"""

import json
import re
from pathlib import Path

###############################################################################
# Globals
###############################################################################

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "single": 1,  # "give a single example" -- same as count=1
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "três": 3,
    "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
}
 
 
def _extract_count(text: str) -> int | None:
    """Pulls a target number out of instruction text -- digit form first
    (most common), falls back to spelled-out EN/PT number words.
 
    Finds the EARLIEST-occurring match by text position, not the first
    match by dict iteration order -- a real bug caught by spot-checking
    real data: "List three numbers between one and ten" was extracting
    1 (from "one", later in the dict lookup order) instead of 3 (the
    actual target count, which appears first in the sentence), since
    the sentence contains multiple number words and the target is
    virtually always the one closest to the action verb, not whichever
    happens to be checked first.
    """
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
 
    earliest_pos, earliest_value = None, None
    for word, n in _NUMBER_WORDS.items():
        match = re.search(rf"\b{word}\b", text, re.IGNORECASE)
        if match and (earliest_pos is None or match.start() < earliest_pos):
            earliest_pos, earliest_value = match.start(), n
    return earliest_value
 
 
PATTERNS = [
    # (regex on the instruction text, constraint_type, value_kind)
    # value_kind: False = no value needed, "quote" = pull from quotes,
    # "count" = pull a target number via _extract_count
    (re.compile(r"\bsingle number\b|\bum n[uú]mero\b|\bum [uú]nico n[uú]mero\b", re.IGNORECASE), "single_number", False),
    (re.compile(r"\blowercase\b|\bmin[uú]sculas\b", re.IGNORECASE), "lowercase", False),
    (re.compile(r"\bone[\s-]word\b|\buma palavra\b", re.IGNORECASE), "one_word", False),
    (re.compile(r"\byes or no\b|\bsim ou n[aã]o\b", re.IGNORECASE), "yes_no", False),
    (re.compile(r"\ball uppercase\b|\bcapital letters\b|\bmai[uú]sculas\b", re.IGNORECASE), "uppercase", False),
    (re.compile(r"\bbackwards\b|\bde tr[aá]s para frente\b", re.IGNORECASE), "reverse_word", "quote"),
    (re.compile(r"\brepeat exactly\b|\brepita exatamente\b|\brepeat the word\b|\bsay the word\b.{0,15}back\b|"
                 r"\brepita a palavra\b|\bdiga a palavra\b.{0,20}volta\b", re.IGNORECASE), "exact_match", "quote"),
    (re.compile(r"\bno more than \w+ words\b|\bat most \w+ words\b|\bfewer than \w+ words\b|\bunder \w+ words\b|"
                 r"\b\w+ words? or fewer\b|\bwords? maximum\b|\bkeep.{0,15}\w+ words\b|"
                 r"\bm[aá]ximo.{0,10}\w+ palavras\b|\baté \w+ palavras\b", re.IGNORECASE), "max_words", "count"),
    (re.compile(r"\bexactly \d+\b|\blist exactly\b|\bname exactly\b|\bgive exactly\b|\bmention exactly\b|"
                 r"\bliste exatamente\b|\bnomeie exatamente\b|\bmencione exatamente\b|\bcite exatamente\b|"
                 r"\bd[êe] exatamente\b|"
                 r"\b(list|name|cite|give( me)?)\s+(a\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|single)\b|"
                 r"\b(liste|cite)\s+(um|uma|dois|duas|tr[eê]s|quatro|cinco|seis|sete|oito|nove|dez|\d+)\b|"
                 r"\bd[êe]\s+(um|uma|\d+)\s+exempl",  # "dê um exemplo de X" -- requires the actual real collocation, not bare d[êe] which also matches the preposition "de"
                 re.IGNORECASE), "exact_item_count", "count"),
    (re.compile(r"\bone sentence\b|\bsingle sentence\b|\buma frase\b|\buma [uú]nica frase\b",
                 re.IGNORECASE), "sentence_count", "count"),
]
 
# For reverse_word / exact_match, tries to pull the target word out of a
# quoted span in the instruction (e.g. "reverse the word 'trombeta'").
# Falls back to None (needs manual entry) if no quote is found.
QUOTE_RE = re.compile(r"['\"]([^'\"]+)['\"]")
 
 
def infer_constraint(instruction_text: str) -> dict | None:
    for pattern, constraint_type, value_kind in PATTERNS:
        if pattern.search(instruction_text):
            result = {"constraint_type": constraint_type}
            if value_kind == "quote":
                match = QUOTE_RE.search(instruction_text)
                result["constraint_value"] = match.group(1) if match else None
            elif value_kind == "count":
                # sentence_count defaults to 1 if no number found ("one
                # sentence" phrasings don't always contain a digit)
                count = _extract_count(instruction_text)
                if count is None and constraint_type == "sentence_count":
                    count = 1
                result["constraint_value"] = count
            return result
    return None
 
 
def tag_file(path: Path) -> tuple[list[dict], list[dict]]:
    """Returns (confidently_tagged, needs_manual_review) -- doesn't write anything."""
    tagged, needs_review = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instruction_text = row["messages"][0]["content"]
            inferred = infer_constraint(instruction_text)
 
            if inferred is None:
                needs_review.append(row)
            elif inferred.get("constraint_value") is None and "constraint_value" in inferred:
                # matched a type that needs a value, but couldn't extract one
                needs_review.append(row)
            else:
                tagged.append({**row, **inferred})
 
    return tagged, needs_review
 
 
def apply_tags(path: Path, output_path: Path, review_output_path: Path) -> None:
    """
    Writes two files:
      - output_path: every confidently-tagged row, with constraint_type
        (and constraint_value where applicable) merged into the row
        alongside its existing fields.
      - review_output_path: rows that need manual review, UNCHANGED --
        so you can hand-edit just this smaller file and re-run, rather
        than re-tagging everything from scratch.
    Never overwrites the original input file.
    """
    tagged, needs_review = tag_file(path)
 
    with open(output_path, "w", encoding="utf-8") as f:
        for row in tagged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
 
    with open(review_output_path, "w", encoding="utf-8") as f:
        for row in needs_review:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
 
    print(f"Wrote {len(tagged)} tagged rows to {output_path}")
    print(f"Wrote {len(needs_review)} rows needing review to {review_output_path}")
 
 
def main():
    # Edit these three paths directly, then just run the file in Spyder --
    # no command-line args needed.
    
    data_paths = Path("data") / "sft" / "conversation" / "level0" 
    tagged_path = Path("runs") / "instruction_following_tag"
    tagged_path.mkdir(parents=True, exist_ok=True)
    
    input_paths = list(data_paths.rglob("*instruction*.jsonl"))
    
    for input_path in input_paths:
        
        save_path = tagged_path / input_path.parts[4] / input_path.parts[5] / input_path.parts[6]
        save_path.mkdir(parents=True, exist_ok=True)
        
        tagged_output_path = save_path / str(input_path.stem + "_taggeg.jsonl")
        review_output_path = save_path / str(input_path.stem + "_needs_review.jsonl")
    

        # Set to True only after you've spot-checked a sample of the tagged
        # output below and trust it. Defaults to False so a first run never
        # accidentally writes files -- it just reports what it WOULD do.
        actually_write_files = False
    
        tagged, needs_review = tag_file(input_path)
    
        print(f"Confidently tagged: {len(tagged)}")
        for r in tagged[:15]:
            print(f"  [{r['id']}] {r['constraint_type']:<14} "
                  f"{r.get('constraint_value','')!r:20} <- {r['messages'][0]['content'][:50]!r}")
    
        print(f"\nNeeds manual review: {len(needs_review)}")
        for r in needs_review[:15]:
            print(f"  [{r['id']}] {r['messages'][0]['content'][:60]!r}")
    
        if actually_write_files:
            apply_tags(input_path, tagged_output_path, review_output_path)
        else:
            print(f"\nactually_write_files is False -- nothing written. "
                  f"Spot-check the tagged sample above, then set it to True and rerun.")


if __name__ == "__main__":
    main()