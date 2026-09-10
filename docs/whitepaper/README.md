# Implementation white paper

`whitepaper.tex` is the operator-level implementation manual for the
VestigeKV attention backend: the selection signal, data structures and
memory contract, the single-request and batched decode paths, the
attention-kernel contract, the seven-kernel fused in-graph recall scan, the
cost model, and the observed defect classes with their mitigations. It is
written for someone porting or maintaining the kernels, not for the paper's
reviewers, and is kept out of the submission's appendix deliberately (the
paper carries the theory, the two-metric results, and the ArkVale
comparison; this is the how-to-build companion).

```bash
pdflatex whitepaper.tex   # self-contained, no \input dependencies
```
