"""Read one dotted key out of the run's result JSON, printing the empty string if absent.

Exists so the job template contains no inline ``python3 -c`` one-liner. TASK-N18 constraint 1 is
about embedded scripts surviving SSM and YAML quoting, and a quoted one-liner is an embedded
script with worse odds than a file.

Usage::

    python3 get.py <result.json> <dotted.key>
"""
import json
import sys

try:
    with open(sys.argv[1]) as fh:
        node = json.load(fh)
except Exception:                                       # noqa: BLE001
    print("")
    raise SystemExit(0)

for part in sys.argv[2].split("."):
    if not isinstance(node, dict) or part not in node:
        print("")
        raise SystemExit(0)
    node = node[part]
print(node if isinstance(node, str) else json.dumps(node))
