# Development

## Package Boundaries

Keep reusable robot logic in `cpsquare-lab` and executable task experiments in this repository. Task-specific geometry, rewards, observations, and replay helpers should stay with the task that owns them.

Do not reintroduce top-level compatibility packages for deprecated `cpsquare_lab.assets`, `cpsquare_lab.utils`, or `cpsquare_lab.mdp` surfaces. Update imports and tests to the current package layout instead.

## Documentation Standards

Python modules, public classes, public methods, and functions are documented with Google-style docstrings. Ruff enforces pydocstyle through the `D` rule family and `tool.ruff.lint.pydocstyle.convention = "google"`.

Use module docstrings to explain purpose and constants. Use inline comments sparingly for non-obvious math, paper-derived coefficients, and simulator-specific workarounds.

## Checks

Run focused checks before handing off changes:

```bash
uv run ruff check environments/environments scripts --select D
uv run pytest
uv run --group docs mkdocs build --strict
```
