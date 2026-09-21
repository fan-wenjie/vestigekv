"""Regenerate the RULER macros of the paper (vestigekv_paper/ruler_numbers.tex)
from the run records under results/.

    python mexp/kimi/make_ruler_numbers.py [--out ~/vestigekv_paper/ruler_numbers.tex]

Kimi Linear: results/kimi/ruler/results_{baseline,vestigekv}_n10_4096-...-65536.json
(and the 128k-1M files once measured; \\PENDING until then). GLM-5.3-Flash (the
DSA-trained contrast): results/glm53/ruler/results_{baseline,vestigekv}_*.json.
The recall-margin sweep comes from results/kimi/ruler_vestigekv_margin-*.log.
"""

import argparse
import glob
import itertools
import json
import math
import os
import random
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
         "niah_multikey_3", "niah_multiquery", "niah_multivalue", "ruler_vt", "ruler_cwe", "ruler_fwe",
         "ruler_qa_squad", "ruler_qa_hotpot"]
SHORT = {"niah_single_1": "NiahSa", "niah_single_2": "NiahSb", "niah_single_3": "NiahSc",
         "niah_multikey_1": "NiahMKa", "niah_multikey_2": "NiahMKb", "niah_multikey_3": "NiahMKc",
         "niah_multiquery": "NiahMQ", "niah_multivalue": "NiahMV", "ruler_vt": "VT", "ruler_cwe": "CWE",
         "ruler_fwe": "FWE", "ruler_qa_squad": "QAsq", "ruler_qa_hotpot": "QAhp"}
LEN = {4096: "FourK", 8192: "EightK", 16384: "SixteenK", 32768: "ThirtyTwoK", 65536: "SixtyFourK",
       131072: "OneTwoEightK", 262144: "TwoFiveSixK", 524288: "FiveTwelveK", 1048576: "OneM"}


def load(line, arm, n, lengths):
    """Exact filename first, then the tagged variant the runner writes.

    A job whose queue entry carries a tag lands as
    results_<arm>_n<N>_<lengths>_<tag>.json, so the untagged path misses it and
    the macros silently come out PENDING -- which is how a finished run can look
    like an unfinished one."""
    base = os.path.join(ROOT, "results", line, "ruler",
                        f"results_{arm}_n{n}_{'-'.join(map(str, lengths))}")
    if os.path.exists(base + ".json"):
        return json.load(open(base + ".json"))
    tagged = sorted(glob.glob(base + "_*.json"))
    if not tagged:
        return None
    if len(tagged) > 1:
        # Picking one alphabetically is how a partial sweep replaces the arm's
        # full run without anything saying so: several of these tags are
        # single-task or resumed runs, and they answer the same macro names.
        raise SystemExit(
            f"ABORT: {len(tagged)} tagged candidates for {arm} n={n}, and no "
            "untagged run to prefer:\n  " + "\n  ".join(os.path.basename(t) for t in tagged)
            + "\nName the one that is the arm's run by removing its tag.")
    print(f"   note: {arm} n={n} read from {os.path.basename(tagged[0])} (tagged)")
    return json.load(open(tagged[0]))


def paired(line, n, lengths):
    """Item-level paired differences, when both arms kept their sample records.

    The two arms answer the *same* generated items under the same seed, so the
    arms are not independent samples and a band built as if they were charges
    the estimate with between-item variance that pairing removes -- on the long
    line that band is 0.059 against a paired 0.035, which is the difference
    between calling the gap noise and resolving it. The paired vector is the
    right object, and it is available only where samples_*.json survives: the
    n=50 grid kept one arm's, so it has none.

    Scores are keyed by length inside each record, with -1.0 marking a length
    the item was not run at."""
    d = os.path.join(ROOT, "results", line, "ruler")
    red = os.path.join(d, "paired_scores.json")
    reduced = json.load(open(red))["scores"] if os.path.exists(red) else {}

    def scores(arm):
        """{task: {"<doc_id>,<length>": score}}, full record or reduction.

        The generations are 188 MB per arm and stay out of results.zip, so the
        archive carries mexp/tools/reduce_ruler_samples.py's projection of them
        instead -- the same per-item scores, three orders of magnitude smaller.
        Prefer the full record when it is on disk (re-running a job writes it),
        fall back to the reduction, so the macros come out the same either way."""
        stem = f"{arm}_n{n}_{'-'.join(map(str, lengths))}"
        p = os.path.join(d, f"samples_{stem}.json")
        if not os.path.exists(p):
            g = sorted(glob.glob(os.path.join(d, f"samples_{stem}_*.json")))
            p = g[0] if g else None
        if p:
            raw = json.load(open(p))
            return {t: {f"{s['doc_id']},{l}": s[str(l)] for s in recs
                        for l in lengths if str(l) in s and s[str(l)] >= 0}
                    for t, recs in raw.items()}
        k = next((k for k in reduced if k.startswith(stem)), None)
        return reduced[k] if k else None

    b, v = scores("baseline"), scores("vestigekv")
    if not b or not v:
        return None
    out = []
    for t in b:
        if t not in v or set(b[t]) != set(v[t]):   # items differ: pairing unsound
            return None
        out += [v[t][k] - b[t][k] for k in sorted(b[t])]
    return out


def signflip_p(d):
    """Two-sided exact sign-flip permutation p-value on the mean difference.

    Under exchangeability of the two arms within an item, each non-zero
    difference is equally likely to carry either sign; zeros contribute nothing
    to the statistic and are dropped, which is what makes the enumeration
    affordable here (8 non-zero differences out of 130 items). Falls back to
    sampling if the enumeration would be large."""
    nz = [x for x in d if x != 0]
    if not nz:
        return 1.0
    s = abs(sum(nz))
    if len(nz) <= 22:
        hit = sum(abs(sum(g * x for g, x in zip(signs, nz))) >= s - 1e-9
                  for signs in itertools.product((1, -1), repeat=len(nz)))
        return hit / 2 ** len(nz)
    rng = random.Random(0)
    trials = 200000
    hit = sum(abs(sum(x if rng.random() < 0.5 else -x for x in nz)) >= s - 1e-9
              for _ in range(trials))
    return hit / trials


def cell(r, t, l):
    row = r["results"].get(t, {})
    v = row.get(f"{l},none", row.get(str(l)))
    return None if v is None else float(v)


def fmt(v):
    return "\\PENDING" if v is None else f"{v:.2f}"


def fmt3(v):
    return "\\PENDING" if v is None else f"{v:.3f}"


def fmtwall(r):
    return "\\PENDING" if r is None else f"{r['wall_s'] / 60:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/vestigekv_paper/ruler_numbers.tex"))
    args = ap.parse_args()
    L = ["% Generated by mexp/kimi/make_ruler_numbers.py from results/. Do not edit by hand."]
    # n is per line, not global: the Kimi grid is measured at 50 per cell (one
    # sample moves a cell by 0.02), the long grid at 5, GLM-5.3 at 10.
    for line, tag, n, lengths in (("kimi", "K", 50, [4096, 8192, 16384, 32768, 65536]),
                                  ("kimi", "KL", 5, [131072, 262144]),   # 512k/1M dropped: see README
                                  # Table 1's Base column, same protocol as the Instruct grid
                                  # above so the two halves of that row compare; separate
                                  # directory because the filenames share arm/n/lengths.
                                  ("kimi_base", "KB", 50,
                                   [4096, 8192, 16384, 32768, 65536]),
                                  ("glm53", "G", 10, [4096, 8192, 16384, 32768, 65536])):
        arms = {"D": load(line, "baseline", n, lengths), "V": load(line, "vestigekv", n, lengths)}
        for a, r in arms.items():
            for t in TASKS:
                vals = [cell(r, t, l) if r else None for l in lengths]
                for l, v in zip(lengths, vals):
                    L.append(f"\\newcommand{{\\ruler{tag}{a}{SHORT[t]}{LEN[l]}}}{{{fmt(v)}}}")
                m = None if any(v is None for v in vals) else sum(vals) / len(vals)
                # fmt3() rather than an inline conditional: a backslash inside
                # an f-string expression is a SyntaxError before Python 3.12,
                # so this one line decided whether a reader on 3.11 could
                # regenerate any of this file's 685 macros.
                L.append(f"\\newcommand{{\\ruler{tag}{a}{SHORT[t]}Mean}}{{{fmt3(m)}}}")
            for l in lengths:
                vals = [cell(r, t, l) if r else None for t in TASKS]
                m = None if any(v is None for v in vals) else sum(vals) / len(vals)
                L.append(f"\\newcommand{{\\ruler{tag}{a}Mean{LEN[l]}}}{{{fmt3(m)}}}")
            allv = [cell(r, t, l) if r else None for t in TASKS for l in lengths]
            m = None if any(v is None for v in allv) else sum(allv) / len(allv)
            L.append(f"\\newcommand{{\\ruler{tag}{a}Mean}}{{{fmt3(m)}}}")
            L.append(f"\\newcommand{{\\ruler{tag}{a}Wall}}{{{fmtwall(r)}}}")
        if arms["D"] and arms["V"]:
            d = [cell(arms["V"], t, l) - cell(arms["D"], t, l) for t in TASKS for l in lengths]
            L.append(f"\\newcommand{{\\ruler{tag}Delta}}{{{sum(d) / len(d):+.3f}}}")
            # Per-task deltas. The limitation section classifies the thirteen
            # tasks by their generator parameters and shows the loss sitting
            # only where the haystack is itself made of key-value pairs, so
            # each delta has to be a macro rather than a number typed into
            # prose that a re-run would silently invalidate.
            for t in TASKS:
                dv = [cell(arms["V"], t, l) - cell(arms["D"], t, l) for l in lengths]
                L.append(f"\\newcommand{{\\ruler{tag}Delta{SHORT[t]}}}"
                         f"{{{sum(dv) / len(dv):+.3f}}}")
            L.append(f"\\newcommand{{\\ruler{tag}Worse}}{{{sum(x < -1e-9 for x in d)}}}")
            L.append(f"\\newcommand{{\\ruler{tag}Better}}{{{sum(x > 1e-9 for x in d)}}}")
            L.append(f"\\newcommand{{\\ruler{tag}Cells}}{{{len(d)}}}")
            # Resolution of the line, as 2 sigma on the mean difference.
            #
            # Two bands, and they are not interchangeable. The *unpaired* one
            # treats each cell as n Bernoulli trials and the arms as independent
            # runs (variance = sum over cells and arms of p(1-p)/n, over
            # cells^2). That is what the line supports when only one arm kept
            # its sample records -- and it is conservative in a direction that
            # flatters this paper, because a band too wide makes a real gap read
            # as noise and a localisation claim read as cleaner than it is.
            #
            # The arms answer the *same* items under the same seed, so where
            # both arms' samples survive the *paired* band is the correct one
            # and is much tighter (0.035 vs 0.059 on the long line). The paired
            # macros carry a P suffix; the body cites those wherever they exist.
            var = sum(p * (1 - p) / n
                      for a in ("D", "V") for t in TASKS for l in lengths
                      for p in (cell(arms[a], t, l),)) / len(d) ** 2
            L.append(f"\\newcommand{{\\ruler{tag}TwoSigma}}{{{2 * math.sqrt(var):.3f}}}")
            pd = paired(line, n, lengths)
            if pd:
                nz = [x for x in pd if x != 0]
                L.append(f"\\newcommand{{\\ruler{tag}PairedN}}{{{len(pd)}}}")
                L.append(f"\\newcommand{{\\ruler{tag}PairedDelta}}"
                         f"{{{statistics.mean(pd):+.3f}}}")
                L.append(f"\\newcommand{{\\ruler{tag}PairedTwoSigma}}"
                         f"{{{2 * statistics.stdev(pd) / math.sqrt(len(pd)):.3f}}}")
                L.append(f"\\newcommand{{\\ruler{tag}PairedP}}"
                         f"{{{signflip_p(pd):.3f}}}")
                L.append(f"\\newcommand{{\\ruler{tag}Disagree}}{{{len(nz)}}}")
                L.append(f"\\newcommand{{\\ruler{tag}DisagreeDown}}"
                         f"{{{sum(x < 0 for x in nz)}}}")
    # The serving telemetry these macros come from lives in the server logs,
    # which are not retained in the archive (see results/kimi/vkstats_extract.json).
    # Prefer the log when it is there -- re-running a job regenerates it -- and
    # fall back to the extract so the macros stay reproducible either way.
    _vk_extract = {}
    _xp = os.path.join(ROOT, "results", "kimi", "vkstats_extract.json")
    if os.path.exists(_xp):
        _vk_extract = {k: v for k, v in json.load(open(_xp)).items()
                       if not k.startswith("_")}

    def vkstats(job):
        s = os.path.join(ROOT, "results", "kimi", f"server_vestigekv_{job}.log")
        if os.path.exists(s):
            last = [l for l in open(s, errors="replace") if "VKSTATS" in l and "TP1]" not in l]
            m = re.search(r"steps=(\d+).*?fetch\[p50=(\d+) p90=(\d+) p99=(\d+)\] fallback=([0-9.]+)",
                          last[-1]) if last else None
            if m:
                return {"steps": int(m.group(1)), "fetch_p50": int(m.group(2)),
                        "fetch_p90": int(m.group(3)), "fetch_p99": int(m.group(4)),
                        "fallback": float(m.group(5))}
        return _vk_extract.get(job)

    # recall-margin sweep (per-job client logs; the targeted tasks at 16k/32k/64k)
    for job, mac in (("margin-0", "Zero"), ("margin-1", "One"), ("margin-2", "Two"), ("margin-3", "Three"),
                     ("margin-lse-2.3", "LseA"), ("margin-lse-4.6", "LseB")):
        p = os.path.join(ROOT, "results", "kimi", f"ruler_vestigekv_{job}.log")
        if not os.path.exists(p):
            continue
        txt = open(p, errors="replace").read()
        cells = {}
        for t in ("niah_single_1", "niah_multikey_2", "niah_multikey_3", "ruler_fwe", "ruler_qa_hotpot"):
            m = re.search(rf"^{t} (\{{.*\}})$", txt, flags=re.M)
            if m:
                d = eval(m.group(1))
                cells[t] = [float(v) for v in d.values()]
        if len(cells) == 5:
            allc = [v for vs in cells.values() for v in vs]
            tgt = [v for t in ("niah_multikey_2", "niah_multikey_3", "ruler_qa_hotpot") for v in cells[t]]
            L.append(f"\\newcommand{{\\sweep{mac}Mean}}{{{sum(allc) / len(allc):.3f}}}")
            L.append(f"\\newcommand{{\\sweep{mac}Targeted}}{{{sum(tgt) / len(tgt):.3f}}}")
            L.append(f"\\newcommand{{\\sweep{mac}MKc}}{{{sum(cells['niah_multikey_3']) / 3:.3f}}}")
            L.append(f"\\newcommand{{\\sweep{mac}QAhp}}{{{sum(cells['ruler_qa_hotpot']) / 3:.3f}}}")
        v = vkstats(job)
        if v:
            L.append(f"\\newcommand{{\\sweep{mac}FetchFifty}}{{{v['fetch_p50']}}}")
            L.append(f"\\newcommand{{\\sweep{mac}Fallback}}{{{v['fallback']:.2f}}}")
    # calibrated-regime stats (stream 128k, stats on) and the 64k short-answer regime
    # Both of these used to name jobs from the retired worktrees, so the two
    # fallback rates the body cites were measured on a tree nothing ships.
    # fbstream-on is the 128k stream with stats on the unified tree; its
    # predecessor reported 0.000 where this one reports 0.012, which is small
    # but is not nothing and the paper said "nothing".
    for job, mac in (("fbstream-on", "StreamOneTwoEight"),
                     ("statsruler64-vestigekv", "RulerSixtyFour")):
        v = vkstats(job)
        # Emit PENDING rather than nothing. A macro that vanishes takes the
        # build with it, and the reflex is then to restore the old value --
        # which here means a number from a retired tree. PENDING compiles and
        # is impossible to mistake for a measurement.
        L.append(f"\\newcommand{{\\stats{mac}Steps}}{{{v['steps'] if v else chr(92) + 'PENDING'}}}")
        L.append(f"\\newcommand{{\\stats{mac}FetchFifty}}{{{v['fetch_p50'] if v else chr(92) + 'PENDING'}}}")
        L.append(f"\\newcommand{{\\stats{mac}FetchNinety}}{{{v['fetch_p90'] if v else chr(92) + 'PENDING'}}}")
        L.append(f"\\newcommand{{\\stats{mac}FetchNinetyNine}}{{{v['fetch_p99'] if v else chr(92) + 'PENDING'}}}")
        L.append(f"\\newcommand{{\\stats{mac}Fallback}}{{{format(v['fallback'], '.3f') if v else chr(92) + 'PENDING'}}}")
    open(args.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {args.out}: {len(L)} macros")


if __name__ == "__main__":
    main()
