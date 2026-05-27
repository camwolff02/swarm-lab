---
name: pre-commit
description: Run IsaacLab pre-commit checks. Use when user asks to run pre-commit, lint, format, ruff, or check code before committing.
---

# Pre-commit

Run IsaacLab's pre-commit checks on all files.

```bash
cd /home/cam/Development/cpsquare/IsaacLab && ./isaaclab.sh -f 2>&1 | tail -5
```

If `isaaclab.sh` is not found, it may need to be run from the IsaacLab repo root. Check with:

```bash
ls /home/cam/Development/cpsquare/IsaacLab/isaaclab.sh
```
