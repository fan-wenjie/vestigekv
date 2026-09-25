#!/usr/bin/env bash
# Static audit of every pre-registered experiment script. Run before running one.
#
#   bash mexp/exp/audit.sh            # audit all
#   bash mexp/exp/audit.sh kvzip*.sh  # audit a subset
#
# It checks the things that have actually gone wrong here, not a generic lint:
# a flag whose units were misread, a budget that OOMs, an output path that
# collides with a record already on disk, a script nobody registered in README,
# and a script that forgot the guards. Every check is a refusal, not a warning:
# an experiment costs GPU hours and a wrong one costs them twice.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
fail=0
note() { echo "  FAIL $1"; fail=$((fail + 1)); }

targets=("${@:-}")
[ -z "${targets[0]:-}" ] && targets=(*.sh)

for f in "${targets[@]}"; do
  case "$f" in _lib.sh | audit.sh) continue ;; esac
  [ -f "$f" ] || continue
  echo "$f"

  grep -q 'source .*_lib.sh\|\. .*_lib.sh' "$f" || note "does not source _lib.sh"
  grep -q 'require_free_gpu' "$f" || note "no require_free_gpu: may interleave with a live job"
  grep -q 'refuse_overwrite' "$f" || note "no refuse_overwrite: may clobber its own record"
  grep -q '^# PREREG:' "$f" || note "no '# PREREG:' line saying what it tests and how to read it"
  bash -n "$f" || note "syntax error"

  # An experiment compares arms, and a comparison is only worth its weakest
  # controlled variable. Every defect this paper has shipped in a comparison
  # was a loose one, not a wrong number: two arms of one row running different
  # operators, an arm at a bandwidth nobody deploys, a control whose engine
  # tree nobody recorded, rows of one table at different trial counts. So a
  # script states what it varies and what it holds, and the statement is
  # required rather than encouraged -- writing it is where the loose variable
  # gets noticed, which is before the GPU hours, not after.
  grep -q '^# CONTROL:' "$f" || note "no '# CONTROL:' block"
  grep -q '^#   varies:' "$f" || note "CONTROL names nothing under 'varies:'"
  grep -q '^#   fixed:' "$f" || note "CONTROL names nothing under 'fixed:'"

  # A fixed seed is the one controlled variable that is invisible in a result
  # file if it was never passed: e2e.py requires --seed, run_ruler.py defaults
  # it, and a default is not a declaration.
  # Comments are stripped first: this check passed on a script whose only
  # --seed was a sentence in its own header, which is the failure mode the
  # check exists to catch.
  if grep -v '^[[:space:]]*#' "$f" | grep -qE 'harness/e2e\.py|run_ruler\.py'; then
    grep -v '^[[:space:]]*#' "$f" | grep -q -- '--seed' \
      || note "invokes a runner without an explicit --seed"
  fi

  # --rhos is a COMPRESSION RATIO (8, 32, 128), parsed as 1/x. A fraction here
  # silently runs a protocol nobody meant; this cost a wasted smoke run.
  for v in $(grep -o -- '--rhos [0-9.,]*' "$f" | awk '{print $2}' | tr ',' ' '); do
    case "$v" in
      *.*) note "--rhos $v is a fraction; the flag wants a ratio (8/32/128)" ;;
      "") ;;
      *) [ "$v" -ge 2 ] 2>/dev/null || note "--rhos $v below 2" ;;
    esac
  done

  # --gpu-expert-layers 99 puts every expert on GPU and OOMs this box; the
  # registered runs use 18.
  for v in $(grep -o -- '--gpu-expert-layers [0-9]*' "$f" | awk '{print $2}'); do
    [ "$v" -le 24 ] || note "--gpu-expert-layers $v will OOM (registered runs use 18)"
  done

  # An --out that already exists means this script cannot run as written.
  for v in $(grep -o -- '--out [^ ]*' "$f" | awk '{print $2}'); do
    case "$v" in *'$'*) continue ;; esac
    [ -e "$ROOT/$v" ] && note "--out $v already exists on disk"
  done

  # The README is the run record; a script it does not mention is not registered.
  base="${f%.sh}"
  grep -q "$base" "$ROOT/README.md" || note "not mentioned in README.md (unregistered)"
done

echo
if [ "$fail" -eq 0 ]; then
  echo "audit: clean"
else
  echo "audit: $fail problem(s); fix before running"
fi
exit "$fail"
