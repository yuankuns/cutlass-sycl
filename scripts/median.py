#!/usr/bin/env python3
"""Median-reduce a `<rep> <key...> device_ms=<ms> TOPS=<t>` log over its reps.

The logs scripts/measure_*.sh produce are one line per (rep, case). A single rep
can land on a contended card, so a mean is not safe: reduce over reps with the
median and print the cases side by side, grouped by their last field.

    scripts/median.py /tmp/tile.log [--pivot]

--pivot puts the last key field (the tile, or diag=N) in columns and everything
before it in rows, with each column's delta against the first column.
"""
import collections
import re
import statistics as st
import sys

LINE = re.compile(r"^rep(\d+)\s+(.*?)\s+device_ms=([\d.]+)\s+TOPS=([\d.]+)\s*$")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pivot = "--pivot" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    runs = collections.defaultdict(list)
    for line in open(args[0]):
        m = LINE.match(line.strip())
        if m:
            runs[m.group(2)].append((float(m.group(3)), float(m.group(4))))
    if not runs:
        print("no `repN <key> device_ms=... TOPS=...` lines found")
        return 1
    med = {k: (st.median(x[0] for x in v), st.median(x[1] for x in v)) for k, v in runs.items()}
    nrep = max(len(v) for v in runs.values())

    if not pivot:
        width = max(len(k) for k in med)
        print(f"{'case':{width}s} {'ms':>8s} {'TOPS':>7s}   (median of {nrep})")
        for k in med:
            print(f"{k:{width}s} {med[k][0]:8.3f} {med[k][1]:7.2f}")
        return 0

    rows, cols = [], []
    for k in med:
        row, _, col = k.rpartition(" ")
        if row not in rows:
            rows.append(row)
        if col not in cols:
            cols.append(col)
    rw = max(len(r) for r in rows)
    print(f"median of {nrep} reps; ms, TOPS, and % against {cols[0]}")
    print(f"{'':{rw}s} " + " ".join(f"{c:>24s}" for c in cols))
    for r in rows:
        base = med.get(f"{r} {cols[0]}")
        line = f"{r:{rw}s} "
        for c in cols:
            v = med.get(f"{r} {c}")
            if v is None:
                line += f"{'-':>24s} "
            elif c == cols[0]:
                line += f"{v[0]:9.3f} {v[1]:6.2f}{'':>8s} "
            else:
                line += f"{v[0]:9.3f} {v[1]:6.2f} {100 * (v[0] / base[0] - 1):+6.1f}% "
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
