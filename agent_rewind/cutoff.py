"""Equality tiers for early cutoff: the hash function decides when a rerun's output counts as unchanged.

exact      Merkle hash of the raw JSON (the store's own hash).
canonical  Merkle hash of the normalised value: sorted keys, whitespace-collapsed and
           case-folded strings, numbers normalised (3.0 == 3, floats to 9 significant digits),
           and fields marked free text in the schema left out.

`free_text` lists field names (last path component) or fnmatch patterns over the full path.
A canonical match that is not a true match is a false cutoff; the audit stage measures them.
"""
from __future__ import annotations

from fnmatch import fnmatch

from .store import BOT, ROOT, View, canon, digest, join, split

FREE = "~free~"


class Exact:
    name = "exact"

    def hash(self, store, branch: str, path: str) -> str:
        return store.hash(branch, path)

    def value(self, path: str, value) -> str:
        return digest("L" + canon(value))


class Canonical:
    name = "canonical"

    def __init__(self, free_text=(), ndigits: int = 9) -> None:
        self.free_text, self.ndigits = tuple(free_text), ndigits

    def is_free(self, path: str) -> bool:
        parts = split(path)
        return bool(parts) and any(parts[-1] == f or fnmatch(path, f) for f in self.free_text)

    def norm(self, value):
        if isinstance(value, bool) or value is None or isinstance(value, int):
            return value
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return str(value)
            return int(value) if value == int(value) else float(f"{value:.{self.ndigits}g}")
        if isinstance(value, str):
            return " ".join(value.split()).casefold()
        if isinstance(value, (list, tuple)):
            return [self.norm(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self.norm(v) for k, v in value.items()}
        return value

    def value(self, path: str, value) -> str:
        return digest("L" + (FREE if self.is_free(path) else canon(self.norm(value))))

    def hash(self, store, branch: str, path: str) -> str:
        view, memo = store.live[branch], {}

        def h(p: str) -> str:
            if p in memo:
                return memo[p]
            if p in view.leaves:
                out = self.value(p, view.leaves[p])
            elif view.kids.get(p):
                out = digest("N" + "|".join(f"{n}:{h(View._child(p, n))}" for n in sorted(view.kids[p])))
            else:
                out = BOT
            memo[p] = out
            return out

        return h(join(split(path)) if split(path) else ROOT)


EXACT = Exact()


def get(cutoff):
    """Resolve a name ('exact' | 'canonical'), an instance, or None (exact)."""
    if cutoff is None or cutoff == "exact":
        return EXACT
    if cutoff == "canonical":
        return Canonical()
    return cutoff
