import json
from pathlib import Path

root = Path(__file__).resolve().parent
pseudos = root / "pseudos"
with open(root / "pseudos.json") as f:
    data = json.load(f)
missing = []
for fname in sorted(set(data.values())):
    if not (pseudos / fname).exists():
        missing.append(fname)
if missing:
    print("Missing pseudopotentials:")
    for name in missing:
        print(f"  {name}")
    raise SystemExit(1)
print("All pseudopotentials found.")
