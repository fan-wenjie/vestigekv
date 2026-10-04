"""Score the LongBench-v1 summarisation predictions already on disk.

Needle retrieval cannot see tier 1: multikey_3 reads 1.000 for every rho from
1/32 to 1/512, and tier 1 ALONE reads 0.000. So a probe that could bound kept
has to need broad context rather than a needle, and long-document
summarisation is the shape on hand -- gov_report, multi_news and qmsum, with
predictions for both arms already generated.

This asks only the prior question: does the probe DISCRIMINATE on this line? If
baseline and vestigekv score the same here, it cannot bound kept either and
spending GPU on a tier-1-only run of it would answer nothing.
"""
import argparse, json, os, sys
from rouge_score import rouge_scorer


def refs_for(task):
    """LongBench answers, from the dataset cache the client used."""
    import glob
    for pat in (f"results/glm53/longbench1/refs_{task}.jsonl",
                f"results/glm53/longbench1/{task}_refs.jsonl"):
        if os.path.exists(pat):
            return {json.loads(l)["_id"]: json.loads(l)["answers"] for l in open(pat)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="gov_report,multi_news,qmsum")
    ap.add_argument("--pair", default=None,
                    help="two paths to pred jsonl files; report their mutual ROUGE-L "
                         "(the same-arm noise floor when both are one arm)")
    a = ap.parse_args()
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    def load(path):
        """The client writes the gold `answers` into each row, so no dataset
        access is needed -- real ROUGE-L and mutual agreement come from the
        same file."""
        out = {}
        for l in open(path):
            r = json.loads(l)
            ans = r.get("answers") or []
            if isinstance(ans, str):
                ans = [ans]
            out[r["_id"]] = (r["pred"], ans)
        return out

    def gold_rouge(d):
        vals = [max(sc.score(g, p)["rougeL"].fmeasure for g in a)
                for p, a in d.values() if a]
        return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)

    if a.pair:
        paths = a.pair.split(",")
        ds = [load(x) for x in paths]
        print(f"{'file':<52}{'n':>5}{'ROUGE-L vs gold':>17}")
        for x, d in zip(paths, ds):
            r, n = gold_rouge(d)
            print(f"  {os.path.basename(os.path.dirname(x)) or '.'}/{os.path.basename(x):<34}"
                  f"{n:>5}{r:>17.4f}")
        if len(ds) == 2:
            common = [i for i in ds[0] if i in ds[1]]
            tot = sum(sc.score(ds[0][i][0], ds[1][i][0])["rougeL"].fmeasure for i in common)
            print(f"\n  mutual agreement over {len(common)} shared ids: "
                  f"{tot/max(len(common),1):.4f}")
        return
    print(f"{'task':<14}{'arm':<12}{'n':>5}{'rougeL':>9}")
    for task in a.tasks.split(","):
        refs = refs_for(task)
        rows = {}
        for arm in ("baseline", "vestigekv"):
            p = f"results/glm53/longbench1/pred_{arm}_{task}.jsonl"
            if not os.path.exists(p):
                continue
            rows[arm] = [json.loads(l) for l in open(p)]
        if refs is None:
            # No references on disk: fall back to arm-vs-arm agreement, which
            # still answers "do the arms differ at all" without a gold set.
            if len(rows) == 2:
                b = {r["_id"]: r["pred"] for r in rows["baseline"]}
                v = {r["_id"]: r["pred"] for r in rows["vestigekv"]}
                common = [i for i in b if i in v]
                agree = sum(sc.score(b[i], v[i])["rougeL"].fmeasure for i in common)
                print(f"{task:<14}{'arm-vs-arm':<12}{len(common):>5}"
                      f"{agree/max(len(common),1):>9.4f}   (no refs on disk;"
                      f" 1.0 would mean identical text)")
            continue
        for arm, rs in rows.items():
            tot = n = 0
            for r in rs:
                gold = refs.get(r["_id"])
                if not gold:
                    continue
                tot += max(sc.score(g, r["pred"])["rougeL"].fmeasure for g in gold)
                n += 1
            if n:
                print(f"{task:<14}{arm:<12}{n:>5}{tot/n:>9.4f}")


if __name__ == "__main__":
    main()
