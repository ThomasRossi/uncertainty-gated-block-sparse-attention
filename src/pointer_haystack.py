"""
pointer_haystack.py
===================

The Pointer-Chase Haystack (PCH) -- a diagnostic long-context benchmark for
*attention selection*, built for the SSA selection study (CLAUDE.md sections
5-8). Name is provisional.

Why a new benchmark
-------------------
Section 7 ran two RULER subtasks and found neither can evaluate a selection
method:
  * niah_multikey -- the needle shares its key word with the query, so a
    content-based selector finds it by surface match. Selection is trivial,
    recall saturates, no policy can differ.
  * vt -- genuinely stresses selection, but Qwen2.5-7B cannot solve it even
    under DENSE attention (~0.33 recall), so the selector's contribution is
    unmeasurable -- there is no ceiling to fall from.

The missing attribute: QUERY-LATENT RELEVANCE
---------------------------------------------
Call a token *query-identifiable* if its relevance can be scored from features
available at selection time -- typically because it shares content with the
query. Call it *query-latent* if its relevance is established only by FIRST
attending to other tokens: the information needed to know it matters has to be
propagated into the residual stream before the selector can see it.

Every production sparse-attention selector -- Quest, H2O, SnapKV, MInference,
NSA -- scores keys against the query or recent state. That is exactly what
niah rewards, and exactly what query-latent relevance defeats. And query-latent
is not a corner case: it is the normal structure of multi-hop reasoning, where
hop k's evidence is only locatable once hop k-1 has been read. Mechanistic
interpretability shows multi-hop is implemented as attention chained ACROSS
LAYERS (stacked attention heads, arXiv:2411.12118; latent multi-hop reasoning,
Yang et al. 2024) -- i.e. along depth, the POMDP time axis of section 4.

Why current benchmarks don't isolate it
----------------------------------------
  * RULER (arXiv:2404.06654), InfiniteBench, HELMET, LongBench measure
    *effective context length* -- model capability. They are tuned so that
    even dense attention fails at long enough context; capability is the
    variable under test. They do not isolate the selector.
  * Sparse-attention papers evaluate their method by end-to-end accuracy on
    those benchmarks. A drop conflates "the selector dropped the token" with
    "the model could not reason." Attention-recall metrics exist, but are
    reported on query-identifiable needles, where they saturate.
  * Multi-hop benchmarks (RULER-vt, BABILong arXiv:2406.10149, and the
    multi-hop QA family -- HotpotQA, 2WikiMultiHop, MuSiQue) test multi-hop
    *reasoning* end-to-end. They are not designed for a reachable dense
    ceiling, so a failure still cannot be attributed to selection.

No standard benchmark dials query-latency, and none is *diagnostic by
construction* -- built so that dense attention scores ~100% (model capability
removed as a confound) and every remaining failure is attributable to the
selector.

The task
--------
A pointer chase. The haystack is a list of indexed entries; each entry either
redirects -- "Entry ABCD: continue at entry WXYZ." -- or terminates -- "Entry
WXYZ: the recorded value is 481922." The query gives a start id; the answer is
the value at the end of its chain. Distractor chains fill the haystack so the
entry FORMAT carries no signal -- only the specific id matches.

  * hop 0 is query-identifiable (the start id is in the query) -- niah-like.
  * hops 1..h are query-LATENT: the id of entry k appears only inside entry
    k-1, so the selector cannot score entry k until entry k-1 has been read
    and propagated. This is the attribute the benchmark isolates.
  * compounding by construction: miss entry k and entries k+1..h become
    permanently unreachable.
  * reachable ceiling: each step is a trivial copy (follow a link, or read a
    value) and the answer is a single number -- a capable model solves it
    under dense attention, so a failure under sparsity is the selector's.

Grounding: the pointer chase is the canonical communication-complexity problem
for unavoidable k-round sequential dependency -- it provably cannot be
shortcut into fewer rounds, which is exactly "k-hop selection with no myopic
shortcut." RULER-vt is its nearest benchmark relative; PCH differs by being
diagnostic -- a single-value answer and a trivial per-hop step keep the dense
ceiling near 100%, instead of being capability-bound.

Knobs: num_hops (depth / compounding), context length, num_distractor_chains
(selection pressure).
"""

from ruler_tasks import (Example, WORDS_PER_TOKEN, _build_haystack,
                         _rand_number, _rand_varname)


def _chain(rng, n_hops, used):
    """A chain of n_hops redirects ending in a recorded value."""
    ids = [_rand_varname(rng, used) for _ in range(n_hops + 1)]
    value = _rand_number(rng, digits=6)
    lines = [f"Entry {ids[k]}: continue at entry {ids[k + 1]}."
             for k in range(n_hops)]
    lines.append(f"Entry {ids[n_hops]}: the recorded value is {value}.")
    return ids, value, lines


def make_pointer_chase(rng, target_tokens, num_hops=3, num_distractor_chains=10):
    """
    One gold chain plus distractor chains, all scattered through the haystack.

    needles is hop-ordered: needles[k] is the entry the model must reach at
    hop k (needles[0..h-1] redirect, needles[h] holds the answer) -- so the
    harness can compute a per-hop selector hit-rate.
    """
    used = set()
    gold_ids, gold_value, gold_lines = _chain(rng, num_hops, used)

    distractor_lines = []
    for _ in range(num_distractor_chains):
        # decoy chains of varied depth, so chain length is not a giveaway
        h = rng.randint(1, max(2, num_hops))
        _, _, lines = _chain(rng, h, used)
        distractor_lines += lines

    units = gold_lines + distractor_lines
    rng.shuffle(units)
    haystack = _build_haystack(rng, int(target_tokens * WORDS_PER_TOKEN), units)

    query = (f"\n\nQuestion: start at entry {gold_ids[0]} and follow each "
             f"'continue at entry' link until you reach an entry with a "
             f"recorded value. What is that recorded value? Answer with the "
             f"number only.")
    instruction = ("Below is a list of indexed entries. Each entry either "
                   "redirects to another entry or states a recorded value.\n\n")

    return Example(
        task="pointer_chase",
        prompt=instruction + haystack + query,
        answer=[gold_value],
        needles=gold_lines,
        distractors=distractor_lines,
        meta=dict(target_tokens=target_tokens, num_hops=num_hops,
                  num_distractor_chains=num_distractor_chains,
                  gold_ids=gold_ids),
    )


if __name__ == "__main__":
    import random
    from ruler_tasks import score
    rng = random.Random(0)
    for hops in (1, 3):
        ex = make_pointer_chase(rng, 2500, num_hops=hops, num_distractor_chains=8)
        print("=" * 72)
        print(f"num_hops={hops}  answer={ex.answer}  gold_ids={ex.meta['gold_ids']}")
        print("gold chain (hop-ordered needles):")
        for k, line in enumerate(ex.needles):
            print(f"  [{k}] {line}")
        print(f"prompt chars={len(ex.prompt)}; tail:")
        print("  ...", ex.prompt[-260:])
        print(f"score(gold)={score(ex, 'the value is ' + ex.answer[0])}  "
              f"score(wrong)={score(ex, 'the value is 000000')}")
