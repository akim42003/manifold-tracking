#!/usr/bin/env python3
"""Vocabulary diagnostic — understand tokenizer composition and filter relaxation options.

For each token in the vocabulary, assigns it to exactly one bucket, then reports
what the anchor-vocab size would be under progressively relaxed filters.

Usage:
    python vocab_diagnostic.py --vindex gemma3-4b.vindex
    python vocab_diagnostic.py --vindex mistral-7b.vindex --examples 20
"""

import argparse
import json
import os
import re
import sys
import unicodedata

SENTINEL = "▁"   # ▁  — SentencePiece word-start marker

# ── character-class helpers ───────────────────────────────────────────────────

def _script(ch):
    """Return the Unicode script block name for a single character."""
    try:
        name = unicodedata.name(ch, "")
    except (ValueError, TypeError):
        return "UNKNOWN"
    # Fast path for common cases
    if "LATIN" in name:   return "LATIN"
    if "CJK" in name:     return "CJK"
    if "CYRILLIC" in name: return "CYRILLIC"
    if "ARABIC" in name:  return "ARABIC"
    if "DEVANAGARI" in name: return "DEVANAGARI"
    if "GREEK" in name:   return "GREEK"
    if "HEBREW" in name:  return "HEBREW"
    if "HANGUL" in name:  return "HANGUL"
    if "HIRAGANA" in name or "KATAKANA" in name: return "JAPANESE"
    return "OTHER"


def is_ascii_alpha(ch):
    return ch.isascii() and ch.isalpha()


def is_latin(ch):
    """True for ASCII letters and Latin-extended letters (diacritics, accents, etc.)."""
    if not ch.isalpha():
        return False
    if ch.isascii():
        return True
    return _script(ch) == "LATIN"


def is_any_alpha(ch):
    return ch.isalpha()


# ── filters (each takes the body = token stripped of leading ▁) ───────────────

RE_ASCII_WORD_3 = re.compile(r"^[A-Za-z]{3,}$")
RE_ASCII_WORD_2 = re.compile(r"^[A-Za-z]{2,}$")


def f_ascii_3(body):   return bool(RE_ASCII_WORD_3.match(body))
def f_ascii_2(body):   return bool(RE_ASCII_WORD_2.match(body))
def f_latin_3(body):   return len(body) >= 3 and all(is_latin(c) for c in body)
def f_alpha_3(body):   return len(body) >= 3 and all(is_any_alpha(c) for c in body)
def f_nonsub_3(body):  return len(body) >= 3   # anything ≥3 chars, no sub-word check needed here


# ── bucket assignment ─────────────────────────────────────────────────────────

def assign_bucket(token, special_ids, tid):
    """Assign token to exactly one bucket. Returns (bucket_name, body)."""
    if tid in special_ids:
        return "special", token

    if not token.startswith(SENTINEL):
        return "sub_word", token

    body = token[len(SENTINEL):]

    if len(body) < 3:
        return "too_short", body

    if f_ascii_3(body):
        return "ascii_latin_word", body

    all_alpha = all(c.isalpha() for c in body)
    has_alpha  = any(c.isalpha() for c in body)

    if all_alpha:
        if all(is_latin(c) for c in body):
            return "latin_word", body          # Latin + diacritics, not pure ASCII
        else:
            return "other_script_word", body   # Cyrillic, Arabic, CJK, etc.
    elif has_alpha:
        return "mixed_content", body           # letters mixed with digits/symbols
    else:
        return "pure_non_alpha", body          # punctuation, numbers, symbols


# ── main ──────────────────────────────────────────────────────────────────────

BUCKET_ORDER = [
    "special",
    "sub_word",
    "too_short",
    "ascii_latin_word",
    "latin_word",
    "other_script_word",
    "mixed_content",
    "pure_non_alpha",
]

BUCKET_DESC = {
    "special":          "Special / added tokens (BOS, EOS, PAD, etc.)",
    "sub_word":         "Sub-word pieces (no ▁ prefix — continuation tokens)",
    "too_short":        "Word-start (▁) but body < 3 chars",
    "ascii_latin_word": "Current filter: ▁ + 3+ ASCII letters  ← anchor baseline",
    "latin_word":       "▁ + 3+ Latin chars including diacritics (café, naïve, Zürich…)",
    "other_script_word":"▁ + 3+ chars, all-alpha, non-Latin script (Cyrillic, Arabic, CJK…)",
    "mixed_content":    "▁ + letters mixed with digits/symbols",
    "pure_non_alpha":   "▁ + no letters at all (numbers, punctuation, symbols)",
}

FILTERS = [
    ("ASCII Latin ≥ 3 chars  [current]",    f_ascii_3),
    ("ASCII Latin ≥ 2 chars",               f_ascii_2),
    ("Latin + diacritics ≥ 3 chars",        f_latin_3),
    ("Any alphabetic script ≥ 3 chars",     f_alpha_3),
    ("Any non-subword ≥ 3 chars",           f_nonsub_3),
]


def main():
    parser = argparse.ArgumentParser(
        description="Vocabulary composition diagnostic and filter-relaxation explorer.")
    parser.add_argument("--vindex", default="gemma3-4b.vindex")
    parser.add_argument("--examples", type=int, default=15,
                        help="Example tokens to show per relaxation layer (default 15)")
    args = parser.parse_args()

    tok_path = os.path.join(args.vindex, "tokenizer.json")
    if not os.path.exists(tok_path):
        sys.exit(f"tokenizer.json not found in {args.vindex}")

    tok_json = json.load(open(tok_path))
    vocab    = tok_json["model"]["vocab"]            # {token: id}
    special_ids = {t["id"] for t in tok_json.get("added_tokens", [])}
    special_toks = {t["content"] for t in tok_json.get("added_tokens", [])}

    total = len(vocab)
    print(f"Vindex       : {args.vindex}")
    print(f"Vocab size   : {total:,}")
    print(f"Special added: {len(special_ids):,}")
    print()

    # ── bucket assignment ──
    buckets = {b: [] for b in BUCKET_ORDER}
    for token, tid in vocab.items():
        bucket, body = assign_bucket(token, special_ids, tid)
        buckets[bucket].append((token, tid, body))

    # ── bucket report ──
    print("── Bucket breakdown ──────────────────────────────────────────────────")
    print(f"  {'Bucket':<22} {'Count':>7}  {'%':>6}  Description")
    print(f"  {'─'*80}")
    for b in BUCKET_ORDER:
        n = len(buckets[b])
        pct = n / total * 100
        print(f"  {b:<22} {n:>7,}  {pct:>5.1f}%  {BUCKET_DESC[b]}")
    print(f"  {'─'*80}")
    print(f"  {'TOTAL':<22} {total:>7,}")
    print()

    # ── sample tokens per bucket (non-trivial ones) ──
    interesting = ["latin_word", "other_script_word", "mixed_content",
                   "too_short", "pure_non_alpha"]
    for b in interesting:
        items = buckets[b]
        sample = [tok for tok, _, _ in items[:args.examples]]
        print(f"── Sample: {b} ({len(items):,} tokens) ──")
        print(f"  {' '.join(repr(t) for t in sample)}")
        print()

    # ── progressive filter relaxation ──
    # Build each filter's full set (from all word-start ▁ tokens, excl. special)
    filter_sets = []
    for label, fn in FILTERS:
        s = {token for token, tid in vocab.items()
             if token not in special_toks
             and token.startswith(SENTINEL)
             and fn(token[len(SENTINEL):])}
        filter_sets.append((label, fn, s))

    print("── Progressive filter relaxation ─────────────────────────────────────")
    print(f"  {'Filter':<45} {'Size':>7}  {'Δ new':>7}")
    print(f"  {'─'*65}")
    prev_set = set()
    newly_sets = []
    for label, fn, s in filter_sets:
        new = s - prev_set
        print(f"  {label:<45} {len(s):>7,}  {len(new):>+7,}")
        newly_sets.append((label, new))
        prev_set = s
    print()

    # ── examples of newly admitted tokens at each relaxation step ──
    for i, (label, new_tokens) in enumerate(newly_sets):
        if i == 0:
            # Baseline: sample from the current filter set
            sample_pool = list(filter_sets[0][2])
        else:
            sample_pool = list(new_tokens)

        if not sample_pool:
            continue

        # Sort for stability; show bodies (strip ▁)
        sample_pool.sort()
        n_show = min(args.examples, len(sample_pool))
        # For relaxation steps, spread across the sorted list so we see variety
        if len(sample_pool) > n_show:
            step = len(sample_pool) // n_show
            sample = [sample_pool[j * step] for j in range(n_show)]
        else:
            sample = sample_pool

        bodies = [t[len(SENTINEL):] for t in sample]

        header = "baseline" if i == 0 else f"newly admitted by step {i+1}"
        print(f"── Examples ({header}): {label} ──")
        print(f"  {' '.join(repr(b) for b in bodies)}")
        if i == 1:
            print(f"  ^ These 2-char tokens are function words / prepositions."
                  f" High count, noisy as anchors.")
        if i == 2:
            # Latin+diacritics step: check character composition
            accented = [b for b in [t[len(SENTINEL):] for t in new_tokens]
                        if any(not c.isascii() for c in b)]
            accented.sort()
            n_acc = min(args.examples, len(accented))
            if len(accented) > n_acc:
                step2 = max(1, len(accented) // n_acc)
                acc_sample = [accented[j * step2] for j in range(n_acc)]
            else:
                acc_sample = accented
            print(f"  Tokens with non-ASCII chars in this step ({len(accented):,} total):")
            print(f"  {' '.join(repr(b) for b in acc_sample)}")
        print()

    # ── recommendation ──
    n_ascii   = len(filter_sets[0][2])
    n_latin   = len(filter_sets[2][2])
    n_alpha   = len(filter_sets[3][2])
    n_new_latin = len(newly_sets[2][1])
    n_new_alpha = len(newly_sets[3][1])

    print("── Recommendation ────────────────────────────────────────────────────")
    print(f"  Current anchor vocab (ASCII ≥3): {n_ascii:,}")
    print(f"  With Latin+diacritics:           {n_latin:,}  (+{n_new_latin:,}, "
          f"{n_new_latin/n_ascii*100:.1f}% increase)")
    print(f"  With any alphabetic script:      {n_alpha:,}  (+{n_new_alpha:,}, "
          f"{n_new_alpha/n_ascii*100:.1f}% increase on top of latin step)")
    print()
    print("  Decision guide:")
    print("  • Latin+diacritics step: safe if examples above look like real words")
    print("    (café, naïve, Zürich). Risky if dominated by standalone accents (é, ö).")
    print("  • Other-script step: adds multilingual anchors but also noise.")
    print("    Recommended only if model is explicitly multilingual (Gemma, Qwen).")
    print("  • 2-char step: high noise (prepositions, articles). Not recommended.")


if __name__ == "__main__":
    main()
