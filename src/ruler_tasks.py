"""
ruler_tasks.py
==============

Self-contained generator for two RULER (arXiv:2404.06654) long-context tasks,
used for the Mac proof-of-concept (CLAUDE.md section 5, Step 1-on-real-data).

  niah_multikey -- N "magic number" needles hidden in a noise haystack;
                   retrieve the value for one queried key. The other needles
                   are distractors. Tests budget contention.
  vt            -- variable tracking: chains of `VAR a = b` assignments
                   scattered through the haystack; given a value, return every
                   variable in its chain. Multi-hop -- this is the task whose
                   structure matches ssa_pomdp_multilayer.py's chained needles
                   (needle j only resolvable once needle j-1 is found).

Faithful to RULER's task definitions, not a verbatim copy of the repo (which
carries CUDA-side deps). Switch to the official RULER repo for the publication
table; this is the de-risking PoC.

Length is targeted in tokens via a rough words/token ratio. The experiment
harness tokenizes the prompt and maps each needle substring to an exact token
span with the tokenizer's offset mapping -- so needle placement here only has
to be approximate.
"""

from dataclasses import dataclass, field
import string

WORDS_PER_TOKEN = 0.75   # rough Qwen ratio for English prose

# Bland noise haystack, in the spirit of RULER's niah_*_1 noise setting.
FILLER = [
    "The grass is green and the wind moves slowly across the open field.",
    "A quiet road runs past the old house near the edge of town.",
    "Morning light spreads over the rooftops while the streets stay calm.",
    "The river bends around the hill and continues toward the distant coast.",
    "Children walk to school along the path beside the wooden fence.",
    "Clouds drift overhead and the afternoon settles into a steady warmth.",
    "The market opens early and the vendors arrange their crates of fruit.",
    "A train passes through the valley and its sound fades into the trees.",
    "The lamp in the window glows long after the rest of the house is dark.",
    "Rain fell during the night and the garden smells of wet soil.",
    "The cat sleeps on the warm stone near the door of the bakery.",
    "Workers repair the bridge while traffic waits patiently on both sides.",
    "The library is silent except for the turning of a single page.",
    "A boat drifts on the lake and the water mirrors the pale sky.",
    "The bell rings twice and the courtyard slowly fills with people.",
    "Snow gathers on the branches and the path grows quiet and white.",
    "The clock on the wall keeps time that nobody seems to notice.",
    "A gentle breeze carries the smell of bread from the corner shop.",
    "The hills roll outward until they meet the line of the horizon.",
    "Lanterns hang along the street and sway a little in the evening air.",
    "The old man feeds the birds and watches the square come to life.",
    "Far away a dog barks once and then the neighborhood is still again.",
    "The fields are bare now and the season turns toward a colder month.",
    "A letter sits unopened on the table beside a cup of cooling tea.",
]

# Common nouns used as the magic-number keys.
KEY_WORDS = [
    "harbor", "compass", "meadow", "anchor", "candle", "willow", "marble",
    "thunder", "saddle", "lantern", "feather", "glacier", "orchard", "pebble",
    "trumpet", "cavern", "ribbon", "beacon", "cobalt", "juniper", "mosaic",
    "quartz", "velvet", "antler", "driftwood", "ember", "garnet", "hollow",
]


@dataclass
class Example:
    task: str
    prompt: str            # the user-message content (no chat template yet)
    answer: list           # gold answer string(s)
    needles: list          # gold substrings, verbatim in `prompt` (for selection analysis)
    distractors: list = field(default_factory=list)  # distractor substrings
    meta: dict = field(default_factory=dict)


def _build_haystack(rng, target_words, units, depth_lo=0.0, depth_hi=0.85):
    """
    Generate filler sentences to ~target_words, then splice the needle `units`
    into distinct slots within [depth_lo, depth_hi] of the haystack.
    """
    filler, wc = [], 0
    while wc < target_words:
        s = rng.choice(FILLER)
        filler.append(s)
        wc += len(s.split())

    lo = int(len(filler) * depth_lo)
    hi = max(lo + len(units), int(len(filler) * depth_hi))
    slots = sorted(rng.sample(range(lo, hi), len(units)))
    for offset, slot in enumerate(slots):
        filler.insert(slot + offset, units[offset])
    return " ".join(filler)


def _rand_number(rng, digits=7):
    return "".join(rng.choice(string.digits) for _ in range(digits))


def _rand_varname(rng, used):
    while True:
        name = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
        if name not in used:
            used.add(name)
            return name


# ---------------------------------------------------------------------------
# niah_multikey
# ---------------------------------------------------------------------------

def make_niah_multikey(rng, target_tokens, num_keys=4):
    """num_keys magic-number needles; one is queried, the rest are distractors."""
    words = rng.sample(KEY_WORDS, num_keys)
    numbers = [_rand_number(rng) for _ in range(num_keys)]
    needles = [f"One of the special magic numbers for {w} is: {numbers[i]}."
               for i, w in enumerate(words)]

    target = rng.randrange(num_keys)
    rng.shuffle(order := list(range(num_keys)))
    units = [needles[i] for i in order]

    haystack = _build_haystack(rng, int(target_tokens * WORDS_PER_TOKEN), units)
    query = (f"\n\nWhat is the special magic number for {words[target]} "
             f"mentioned in the text above? Answer with the number only.")
    instruction = ("Some special magic numbers are hidden in the following "
                   "text. Memorize them.\n\n")

    return Example(
        task="niah_multikey",
        prompt=instruction + haystack + query,
        answer=[numbers[target]],
        needles=[needles[target]],
        distractors=[needles[i] for i in range(num_keys) if i != target],
        meta=dict(target_tokens=target_tokens, num_keys=num_keys),
    )


# ---------------------------------------------------------------------------
# vt -- variable tracking (multi-hop)
# ---------------------------------------------------------------------------

def _make_chain(rng, num_hops, used_names):
    """A chain of num_hops+1 variables: first holds a value, rest alias it."""
    names = [_rand_varname(rng, used_names) for _ in range(num_hops + 1)]
    value = _rand_number(rng, digits=6)
    lines = [f"VAR {names[0]} = {value}."]
    for i in range(1, len(names)):
        lines.append(f"VAR {names[i]} = {names[i - 1]}.")
    return names, value, lines


def make_vt(rng, target_tokens, num_hops=3, num_distractor_chains=3):
    """
    One gold chain plus distractor chains, all scattered through the haystack.
    The query gives the gold value; the answer is every variable in its chain.
    Multi-hop: the assignment lines appear in random text order, so resolving
    the chain requires tracing, not reading top-to-bottom.
    """
    used = set()
    gold_names, gold_value, gold_lines = _make_chain(rng, num_hops, used)

    distractor_lines = []
    for _ in range(num_distractor_chains):
        _, _, lines = _make_chain(rng, num_hops, used)
        distractor_lines += lines

    units = gold_lines + distractor_lines
    rng.shuffle(units)

    haystack = _build_haystack(rng, int(target_tokens * WORDS_PER_TOKEN), units)
    query = (f"\n\nQuestion: find all variables that are assigned the value "
             f"{gold_value} in the text above (directly or through a chain of "
             f"assignments). Answer with the variable names separated by spaces.")
    instruction = ("Track the chains of variable assignments hidden in the "
                   "following text.\n\n")

    return Example(
        task="vt",
        prompt=instruction + haystack + query,
        answer=gold_names,
        needles=gold_lines,
        distractors=distractor_lines,
        meta=dict(target_tokens=target_tokens, num_hops=num_hops,
                  num_distractor_chains=num_distractor_chains),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(example, model_output):
    """
    niah_multikey / pointer_chase -- 1.0 iff the gold value appears in output.
    vt                            -- recall: fraction of gold variables named.
    musique / narrativeqa / qasper -- SQuAD-style F1 against gold answers
                                     (see longbench_tasks.score for details).
    """
    if example.task in ("musique", "narrativeqa", "qasper", "longbench_v2"):
        import longbench_tasks as lb
        return lb.score(example, model_output)
    out = model_output.upper()
    if example.task in ("niah_multikey", "pointer_chase"):
        return float(example.answer[0] in model_output)
    hits = sum(1 for name in example.answer if name.upper() in out)
    return hits / len(example.answer)


if __name__ == "__main__":
    import random
    rng = random.Random(0)
    for ex in (make_niah_multikey(rng, 2000, num_keys=4),
               make_vt(rng, 2000, num_hops=3, num_distractor_chains=3)):
        print("=" * 70)
        print(f"task={ex.task}  answer={ex.answer}")
        print(f"needles={ex.needles}")
        print(f"prompt chars={len(ex.prompt)}  (head/tail):")
        print(ex.prompt[:240], "...")
        print("...", ex.prompt[-240:])
        print(f"score(gold)={score(ex, ' '.join(ex.answer))}  "
              f"score(empty)={score(ex, '')}")
