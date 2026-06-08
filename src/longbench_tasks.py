"""
longbench_tasks.py
==================

LongBench task loaders + scoring, for the §7.7 LongBench transfer test.

Tasks
-----
- musique   : multi-hop QA, hop-filtered (we load from the original MuSiQue
              dataset because LongBench's MuSiQue subset does not preserve the
              `question_decomposition` length needed for hop stratification.
              Prompt template is LongBench's; scoring is LongBench's F1).
- narrativeqa : single-document narrative QA (LongBench's subset).
- qasper      : single-document scientific QA (LongBench's subset).

Conventions
-----------
- Each `make_example_from_*` returns a `ruler_tasks.Example` so the existing
  `poc_core.run_sweep` harness works unchanged.
- Scoring (`score`) returns SQuAD-style token-overlap F1 against the best of
  the gold answers, a float in [0, 1]. Paired preservation binarises at
  F1 >= F1_THRESHOLD (default 0.5). For PCH the score is exact-match (already
  binary), so threshold=1.0 there.

This file does NOT run anything; it is a pure library of loaders + scorers.
"""

import random
import re
import string
from collections import Counter

from ruler_tasks import Example


F1_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Official LongBench prompt templates (verbatim from THUDM/LongBench).
# ---------------------------------------------------------------------------

MUSIQUE_TEMPLATE = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\n"
    "Question: {input}\n"
    "Answer:"
)

NARRATIVEQA_TEMPLATE = (
    "You are given a story, which can be either a novel or a movie script, "
    "and a question. Answer the question as concisely as you can, using a "
    "single phrase if possible. Do not provide any explanation.\n\n"
    "Story: {context}\n\n"
    "Now, answer the question based on the story as concisely as you can, "
    "using a single phrase if possible. Do not provide any explanation.\n\n"
    "Question: {input}\n\n"
    "Answer:"
)

# LongBench-v2 (THUDM/LongBench-v2, arXiv:2412.15204): 503 multiple-choice
# items, contexts up to 2M words. Uses the official non-CoT eval template --
# we generate at most a handful of tokens and extract the first A/B/C/D.
LONGBENCH_V2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "The correct answer is"
)


QASPER_TEMPLATE = (
    "You are given a scientific article and a question. Answer the question "
    "as concisely as you can, using a single phrase or sentence if possible. "
    "If the question cannot be answered based on the information in the "
    'article, write "unanswerable". If the question is a yes/no question, '
    'answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n'
    "Article: {context}\n\n"
    "Answer the question based on the above article as concisely as you can, "
    "using a single phrase or sentence if possible. If the question cannot be "
    'answered based on the information in the article, write "unanswerable". '
    'If the question is a yes/no question, answer "yes", "no", or '
    '"unanswerable". Do not provide any explanation.\n\n'
    "Question: {input}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_musique(num_hops, max_examples=200, seed=42):
    """Load MuSiQue filtered to `num_hops` (2, 3, or 4). Uses the original
    MuSiQue dataset (dgslibisey/MuSiQue) for hop labels, then applies the
    LongBench prompt template to keep the comparison comparable.

    Returns a list of (task, kw, Example) tuples ready for run_sweep.
    """
    from datasets import load_dataset
    # Validation split has decomposition labels; train has them too but
    # validation is the standard MuSiQue eval.
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    filtered = [
        x for x in ds
        if len(x.get("question_decomposition", [])) == num_hops
        and x.get("answerable", True)            # MuSiQue has unanswerable items
    ]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    filtered = filtered[:max_examples]
    out = []
    for item in filtered:
        ex = _make_musique_example(item)
        out.append(("musique", {"num_hops": num_hops}, ex))
    return out


def load_narrativeqa(max_examples=200, seed=42):
    """Load LongBench's NarrativeQA subset (single-doc QA, negative control)."""
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test",
                      trust_remote_code=True)
    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)
    items = items[:max_examples]
    out = []
    for item in items:
        ex = _make_lb_example(item, "narrativeqa", NARRATIVEQA_TEMPLATE)
        out.append(("narrativeqa", {}, ex))
    return out


def load_longbench_v2(length="medium", max_examples=200, seed=42,
                      difficulty=None, domain=None):
    """Load LongBench-v2 items, optionally filtered by length / difficulty /
    domain. Returns a list of (task, kw, Example) tuples.

    Args:
      length     : "short" | "medium" | "long" | None (no filter). Field is
                   provided per item in the dataset; bucket boundaries are
                   ~32K / ~128K words per the paper.
      difficulty : "easy" | "hard" | None.
      domain     : exact-match on the `domain` field, or None.
    """
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    items = [
        x for x in ds
        if (length is None or x.get("length") == length)
        and (difficulty is None or x.get("difficulty") == difficulty)
        and (domain is None or x.get("domain") == domain)
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:max_examples]
    out = []
    for item in items:
        ex = _make_longbench_v2_example(item)
        kw = {"length": length}
        if difficulty is not None:
            kw["difficulty"] = difficulty
        if domain is not None:
            kw["domain"] = domain
        out.append(("longbench_v2", kw, ex))
    return out


def _make_longbench_v2_example(item):
    """Build an Example from a LongBench-v2 raw item.

    LB2 fields: _id, domain, sub_domain, difficulty, length, question,
    choice_A..D, answer (one of "A","B","C","D"), context.
    """
    prompt = LONGBENCH_V2_TEMPLATE.format(
        context=item["context"],
        question=item["question"],
        choice_A=item["choice_A"],
        choice_B=item["choice_B"],
        choice_C=item["choice_C"],
        choice_D=item["choice_D"],
    )
    return Example(
        task="longbench_v2",
        prompt=prompt,
        answer=[item["answer"]],          # "A" / "B" / "C" / "D"
        needles=[],                       # no gold-span annotations
        meta={
            "_id": item.get("_id", ""),
            "domain": item.get("domain", ""),
            "sub_domain": item.get("sub_domain", ""),
            "difficulty": item.get("difficulty", ""),
            "length": item.get("length", ""),
        },
    )


def load_qasper(max_examples=200, seed=42):
    """Load LongBench's QASPER subset (single-doc scientific QA, neg. control)."""
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench", "qasper", split="test",
                      trust_remote_code=True)
    rng = random.Random(seed)
    items = list(ds)
    rng.shuffle(items)
    items = items[:max_examples]
    out = []
    for item in items:
        ex = _make_lb_example(item, "qasper", QASPER_TEMPLATE)
        out.append(("qasper", {}, ex))
    return out


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def _make_musique_example(item):
    """Build an Example from a raw MuSiQue item.

    MuSiQue fields (validation split):
      - 'paragraphs'             : list of {'idx', 'title', 'paragraph_text',
                                            'is_supporting'}
      - 'question'               : the multi-hop question
      - 'answer'                 : gold answer string
      - 'answer_aliases'         : (optional) list of accepted aliases
      - 'question_decomposition' : list of sub-question/answer/support entries
    """
    paragraphs = item["paragraphs"]
    context = "\n\n".join(
        f"Title: {p.get('title','').strip()}\n{p['paragraph_text'].strip()}"
        for p in paragraphs
    )
    question = item["question"]
    prompt = MUSIQUE_TEMPLATE.format(context=context, input=question)

    answers = [item["answer"]]
    aliases = item.get("answer_aliases") or []
    answers += [a for a in aliases if a not in answers]

    # Supporting paragraphs as `needles` -- they're the gold context blocks
    # the selector should retain. Empty if labels are missing.
    needles = [
        p["paragraph_text"].strip()
        for p in paragraphs if p.get("is_supporting", False)
    ]
    return Example(
        task="musique",
        prompt=prompt,
        answer=answers,
        needles=needles,
        meta={
            "num_hops": len(item.get("question_decomposition", [])),
            "id": item.get("id", ""),
        },
    )


def _make_lb_example(item, task_name, template):
    """LongBench v1 item fields: `input`, `context`, `answers`, `length`,
    `dataset`, `language`."""
    prompt = template.format(context=item["context"], input=item["input"])
    return Example(
        task=task_name,
        prompt=prompt,
        answer=list(item["answers"]),
        needles=[],                  # LongBench doesn't expose gold spans
        meta={"length": int(item.get("length", 0))},
    )


# ---------------------------------------------------------------------------
# Truncation -- LongBench's middle-truncation, applied AFTER loading and
# BEFORE running, with the model's tokenizer.
# ---------------------------------------------------------------------------

def truncate_examples(examples, tokenizer, max_prompt_tokens):
    """In-place: truncate each `ex.prompt` from the middle if its tokenized
    length exceeds `max_prompt_tokens`. Keeps head + tail; drops the middle.
    Matches LongBench's official `truncate_input(manner='middle')`."""
    for _, _, ex in examples:
        enc = tokenizer(ex.prompt, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) <= max_prompt_tokens:
            continue
        half = max_prompt_tokens // 2
        head = tokenizer.decode(ids[:half], skip_special_tokens=True)
        tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
        ex.prompt = head + tail
        # Drop needles -- the original-text matching would be unreliable
        # post-truncation. needle_hit_rate gracefully handles an empty list.
        ex.needles = []


# ---------------------------------------------------------------------------
# Scoring -- SQuAD-style normalised token-overlap F1 (LongBench's choice).
# ---------------------------------------------------------------------------

def _normalize_answer(s):
    """SQuAD normalisation: lower, strip articles + punctuation + extra ws."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _token_f1(prediction, ground_truth):
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def score_mc(example, model_output):
    """Multiple-choice scoring for LongBench-v2: 1.0 if the model's first
    A/B/C/D letter (in 'The correct answer is X' position) matches gold,
    else 0.0. Tolerates 'A.', '(A)', 'A)', or just 'A' patterns."""
    if not example.answer:
        return 0.0
    gold = example.answer[0].strip().upper()
    m = re.search(r"\b([ABCD])\b", model_output.upper())
    if m is None:
        return 0.0
    return float(m.group(1) == gold)


def score(example, model_output):
    """SQuAD-style F1 against the best of the gold answers; float in [0, 1].
    Callers binarise via threshold (default F1_THRESHOLD = 0.5).
    LongBench-v2 is multiple-choice and uses score_mc instead."""
    if example.task == "longbench_v2":
        return score_mc(example, model_output)
    if not example.answer:
        return 0.0
    return max(_token_f1(model_output, gt) for gt in example.answer)
