"""Merge one key into the run's result JSON, keeping JSON-typed values structured.

Carried over from the scratch YAMLs that ran Phase 0-2, where it was the mechanism that made a
preempted run recoverable: the artifact is rewritten in full after every key, so whatever the
last flush uploaded is a valid document rather than a truncated one.

Dotted keys nest (``stages.identity.status``), which is new here -- the scratch version was flat
and every stage's keys collided in one namespace.

Usage::

    python3 rec.py <result.json> <dotted.key> <value>
"""
import json
import sys

path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    with open(path) as fh:
        doc = json.load(fh)
except Exception:                                       # noqa: BLE001
    doc = {}
if not isinstance(doc, dict):
    doc = {}

# Values arrive as text from shell. Keep JSON-typed ones structured so that a nested artifact
# (the C5 arm's whole result document, for instance) survives as an object and not as a string.
try:
    parsed = json.loads(val)
except Exception:                                       # noqa: BLE001
    parsed = val

node = doc
parts = key.split(".")
for part in parts[:-1]:
    nxt = node.get(part)
    if not isinstance(nxt, dict):
        nxt = {}
        node[part] = nxt
    node = nxt
node[parts[-1]] = parsed

with open(path, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
