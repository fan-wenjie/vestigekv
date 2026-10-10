# Installing this skill on another machine

```bash
mkdir -p ~/.claude/skills
unzip glm53-line-resume-skill.zip -d ~/.claude/skills/
chmod +x ~/.claude/skills/glm53-line-resume/scripts/*.sh
```

Then start a session in any directory — the skill is user-level, so it is
discoverable regardless of working directory, and `/glm53-line-resume` invokes it.

## What to do next, in order

1. `reference/state.md` — read first. The settled numbers and the six things not
   to redo. This is what stops a cold session repeating refuted work.
2. `bash scripts/bootstrap.sh` — rebuilds the three checkouts at the pinned SHAs.
   Asks once, never force-updates, refuses a path that already exists.
3. `reference/environment.md` — conda env and `hf download` (~302 G for all three
   models; GLM-5.3-Flash-NVFP4 alone is 191 G and is the only one the line needs).
4. `bash scripts/verify_env.sh` — must print `PREFLIGHT OK` before any number from
   this machine is trusted.

## Provenance

Canonical, version-controlled copy:

    github.com/fan-wenjie/vestigekv   branch glm53-line
    .claude/skills/glm53-line-resume/

This zip is a derived artifact. Fix the repo copy and rebuild the zip; a fix made
only in an unpacked copy is lost at the next machine.
