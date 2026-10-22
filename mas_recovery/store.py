"""Namespace store: one path tree, Merkle hashes, MVCC versions, branches.

Paths look like `/msg/<agent>/<step>/<field>`, `/kv/...`, `/fs/...`, `/world/...`,
`/audit/...`. A JSON value written at a path becomes a subtree (objects and arrays
are inner nodes, everything else is a leaf). The hash of a leaf is the hash of its
canonical JSON; the hash of an inner node covers its children's names and hashes;
an absent path hashes to `BOT`. Every write appends to a log and never overwrites,
so any past state can be rebuilt; a branch is a fork of that log.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

BOT = "⊥"          # hash of an absent path
ROOT = "/"


def canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def split(path: str) -> list[str]:
    return [p for p in path.split("/") if p]


def join(parts: list[str]) -> str:
    return "/" + "/".join(parts)


def cut(path: str, depth: int | None = None) -> str:
    """Truncate `path` to its first `depth` components (None: keep the full path)."""
    parts = split(path)
    return join(parts if depth is None else parts[:depth])


def overlaps(a: str, b: str) -> bool:
    """True when one path is the other or lies below it."""
    pa, pb = split(a), split(b)
    n = min(len(pa), len(pb))
    return pa[:n] == pb[:n]


class View:
    """The materialised tree at one point of one branch."""

    def __init__(self) -> None:
        self.leaves: dict[str, object] = {}
        self.kind: dict[str, str] = {}               # inner node -> "o" (object) | "a" (array)
        self.kids: dict[str, set[str]] = {ROOT: set()}
        self._h: dict[str, str] = {}

    def copy(self) -> "View":
        v = View()
        v.leaves, v.kind = dict(self.leaves), dict(self.kind)
        v.kids = {k: set(s) for k, s in self.kids.items()}
        v._h = dict(self._h)
        return v

    @staticmethod
    def _child(path: str, name: str) -> str:
        return ("" if path == ROOT else path) + "/" + name

    @staticmethod
    def _parent(path: str) -> str:
        parts = split(path)
        return join(parts[:-1]) if len(parts) > 1 else ROOT

    def _dirty(self, path: str) -> None:
        while True:
            self._h.pop(path, None)
            if path == ROOT:
                return
            path = self._parent(path)

    def _drop(self, path: str) -> None:
        for name in list(self.kids.get(path, ())):
            self._drop(self._child(path, name))
        self.kids.pop(path, None)
        self.kind.pop(path, None)
        self.leaves.pop(path, None)
        self._h.pop(path, None)

    def delete(self, path: str) -> None:
        path = join(split(path))
        if path == ROOT:
            self.__init__()
            return
        parent = self._parent(path)
        self._dirty(path)
        self._drop(path)
        self.kids.get(parent, set()).discard(split(path)[-1])
        # prune inner nodes left without children
        while parent != ROOT and not self.kids.get(parent):
            grand = self._parent(parent)
            self._drop(parent)
            self.kids.get(grand, set()).discard(split(parent)[-1])
            parent = grand

    def put(self, path: str, value) -> None:
        path = join(split(path))
        self.delete(path)
        parts = split(path)
        for i in range(len(parts)):                  # make ancestors inner nodes
            anc = join(parts[:i]) if i else ROOT
            self.leaves.pop(anc, None)
            self.kind.setdefault(anc, "o") if anc != ROOT else None
            self.kids.setdefault(anc, set()).add(parts[i])
        self._dirty(path)
        self._insert(path, value)

    def _insert(self, path: str, value) -> None:
        if isinstance(value, dict) and value:
            self.kind[path], self.kids[path] = "o", set()
            for k, v in value.items():
                self.kids[path].add(str(k))
                self._insert(self._child(path, str(k)), v)
        elif isinstance(value, (list, tuple)) and value:
            self.kind[path], self.kids[path] = "a", set()
            for i, v in enumerate(value):
                self.kids[path].add(str(i))
                self._insert(self._child(path, str(i)), v)
        else:
            self.leaves[path] = value if not isinstance(value, tuple) else list(value)

    def hash(self, path: str) -> str:
        path = join(split(path))
        if path in self._h:
            return self._h[path]
        if path in self.leaves:
            h = digest("L" + canon(self.leaves[path]))
        elif self.kids.get(path):
            h = digest("N" + "|".join(f"{n}:{self.hash(self._child(path, n))}" for n in sorted(self.kids[path])))
        else:
            return BOT
        self._h[path] = h
        return h

    def has(self, path: str) -> bool:
        return self.hash(path) != BOT

    def get(self, path: str, default=None):
        path = join(split(path))
        if path in self.leaves:
            return self.leaves[path]
        if not self.kids.get(path):
            return default
        names = sorted(self.kids[path], key=lambda n: (len(n), n))
        sub = {n: self.get(self._child(path, n)) for n in names}
        if self.kind.get(path) == "a":
            return [sub[n] for n in sorted(sub, key=int)]
        return sub

    def subtree_leaves(self, path: str = ROOT) -> dict[str, object]:
        return {p: v for p, v in sorted(self.leaves.items()) if overlaps(p, path) and len(split(p)) >= len(split(path))}


class Store:
    def __init__(self) -> None:
        self.log: list[tuple[int, str, str, str, object]] = []   # (seq, branch, op, path, value)
        self.branches: dict[str, tuple[str | None, int]] = {"main": (None, 0)}
        self.live: dict[str, View] = {"main": View()}
        self.accesses: list[tuple[str, str, list[str], list[str]]] = []   # (branch, step, reads, writes)
        self.seq = 0
        self._n = 0

    # ---- branches ---------------------------------------------------------------
    def new_branch(self, name: str | None = None, parent: str | None = None, at: int = 0) -> str:
        """Fork `parent` as of sequence `at`. The default is an empty branch, which is the
        starting state of a re-execution: it rebuilds state as steps are reused or rerun."""
        if name is None:
            self._n += 1
            name = f"r{self._n}"
        if name in self.branches:
            raise ValueError(f"branch exists: {name}")
        if parent is not None and at == 0:
            at = self.seq
        self.branches[name] = (parent, at)
        self.live[name] = self.snapshot(parent, at) if parent is not None else View()
        return name

    def _chain(self, branch: str, at: int | None) -> list[tuple[str, int]]:
        chain, cur, limit = [], branch, at if at is not None else self.seq
        while cur is not None:
            chain.append((cur, limit))
            parent, fork = self.branches[cur]
            cur, limit = parent, min(fork, limit)
        return chain[::-1]

    def snapshot(self, branch: str, at: int | None = None) -> View:
        if at is None:
            return self.live[branch]
        chain, v = self._chain(branch, at), View()
        for name, limit in chain:
            for seq, b, op, path, value in self.log:
                if b == name and seq <= limit:
                    v.put(path, value) if op == "put" else v.delete(path)
        return v

    # ---- writes and reads ---------------------------------------------------------
    def write(self, branch: str, path: str, value) -> int:
        self.seq += 1
        self.log.append((self.seq, branch, "put", join(split(path)), value))
        self.live[branch].put(path, value)
        return self.seq

    def delete(self, branch: str, path: str) -> int:
        self.seq += 1
        self.log.append((self.seq, branch, "del", join(split(path)), None))
        self.live[branch].delete(path)
        return self.seq

    def hash(self, branch: str, path: str, at: int | None = None) -> str:
        return self.snapshot(branch, at).hash(path)

    def get(self, branch: str, path: str, default=None, at: int | None = None):
        return self.snapshot(branch, at).get(path, default)

    def cut(self, branch: str, path: str, depth: int | None = None, at: int | None = None) -> tuple[str, str]:
        """The recorded form of a read: (cut path, Merkle hash of the cut path)."""
        c = cut(path, depth)
        return c, self.hash(branch, c, at)

    def leaves(self, branch: str) -> dict[str, object]:
        return dict(sorted(self.live[branch].leaves.items()))

    def diff(self, a: str, b: str) -> list[str]:
        """Paths at which two branches differ, found by walking down where hashes differ."""
        va, vb, out = self.live[a], self.live[b], []

        def walk(path: str) -> None:
            if va.hash(path) == vb.hash(path):
                return
            kids = va.kids.get(path, set()) | vb.kids.get(path, set())
            if (path in va.leaves or path in vb.leaves or not kids):
                out.append(path)
                return
            for n in sorted(kids):
                walk(View._child(path, n))

        walk(ROOT)
        return out

    # ---- provenance ---------------------------------------------------------------
    def note(self, branch: str, step: str, reads: list[str], writes: list[str]) -> None:
        self.accesses.append((branch, step, list(reads), list(writes)))

    def forward_cone(self, branch: str, paths) -> list[str]:
        """Steps that read, directly or through other steps' writes, any of `paths`."""
        tainted, hit = set(paths), []
        for b, step, reads, writes in self.accesses:
            if b == branch and any(overlaps(r, t) for r in reads for t in tainted):
                hit.append(step)
                tainted |= set(writes)
        return hit

    def backward_cone(self, branch: str, path: str) -> tuple[list[str], set[str]]:
        """Steps whose output `path` depends on, and the paths they read."""
        needed, steps = {path}, []
        for b, step, reads, writes in reversed(self.accesses):
            if b == branch and any(overlaps(w, n) for w in writes for n in needed):
                steps.append(step)
                needed |= set(reads)
        return steps[::-1], needed

    # ---- persistence --------------------------------------------------------------
    def save(self, path) -> None:
        import os
        if os.path.exists(path):
            os.remove(path)
        db = sqlite3.connect(path)
        db.executescript("""
            create table branches(name text primary key, parent text, fork integer);
            create table log(seq integer primary key, branch text, op text, path text, value text);
            create table accesses(i integer primary key, branch text, step text, reads text, writes text);
        """)
        db.executemany("insert into branches values(?,?,?)", [(n, p, f) for n, (p, f) in self.branches.items()])
        db.executemany("insert into log values(?,?,?,?,?)",
                       [(s, b, o, p, json.dumps(v, default=str)) for s, b, o, p, v in self.log])
        db.executemany("insert into accesses(branch,step,reads,writes) values(?,?,?,?)",
                       [(b, s, json.dumps(r), json.dumps(w)) for b, s, r, w in self.accesses])
        db.commit()
        db.close()

    @classmethod
    def load(cls, path) -> "Store":
        db = sqlite3.connect(path)
        st = cls()
        st.branches = {n: (p, f) for n, p, f in db.execute("select name,parent,fork from branches order by rowid")}
        st.log = [(s, b, o, p, json.loads(v)) for s, b, o, p, v in db.execute("select * from log order by seq")]
        st.accesses = [(b, s, json.loads(r), json.loads(w))
                       for b, s, r, w in db.execute("select branch,step,reads,writes from accesses order by i")]
        db.close()
        st.seq = st.log[-1][0] if st.log else 0
        st._n = len(st.branches)
        st.live = {n: st.snapshot(n, st.seq) for n in st.branches}
        return st
