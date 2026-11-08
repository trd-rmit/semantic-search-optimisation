"""Chunkers return (start, end) char spans so relevance can be judged independent of chunking."""
import re


def fixed(text, size=500, overlap=50):
    # shortcut: sizes are characters, not tokens; switch to a tokenizer if token budgets matter.
    assert 0 <= overlap < size, "overlap must be smaller than size"
    step = size - overlap
    return [(i, min(i + size, len(text))) for i in range(0, max(len(text) - overlap, 1), step) if text[i:i + size].strip()]


def paragraph(text, max_size=1000):
    """Split on blank lines, then greedily merge neighbouring paragraphs up to max_size chars."""
    out = []
    for m in re.finditer(r"\S(?:[^\n]|\n(?!\s*\n))*", text):
        s, e = m.start(), m.end()
        if out and e - out[-1][0] <= max_size:
            out[-1] = (out[-1][0], e)
        elif e - s > max_size:
            out += [(s + a, s + b) for a, b in fixed(text[s:e], max_size, 0)]
        else:
            out.append((s, e))
    return out
