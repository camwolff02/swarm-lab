## Repository Rules

- Do not create or restore a top-level `src/cpsquare_lab/assets` package.
- Do not create or restore a top-level `src/cpsquare_lab/utils` package.
- Robot-specific assets belong under `src/cpsquare_lab/embodiments/`.
- Shared robot-generic code belongs under `src/cpsquare_lab/embodiments/common/`.
- Shared multirotor code belongs under `src/cpsquare_lab/embodiments/multirotor/common/`.
- Robot-specific logic belongs in that robot's embodiment subtree, for example `src/cpsquare_lab/embodiments/multirotor/cf2x/`.
- Do not create or restore a top-level `src/cpsquare_lab/mdp` package.
- Action terms belong with embodiments:
  `embodiments/common/` if robot-generic,
  `embodiments/multirotor/common/` if multirotor-generic,
  or the specific embodiment if not generic.
- Observations belong with the embodiment when they are embodiment-specific, or alongside the task when they are task-specific.
- Rewards belong alongside the task when they are task-specific.
- Task-specific geometry/grid helpers such as `grid_sdf` belong with the relevant task, not in a shared `utils` package.
- Terminations and task config should live with the task package they serve.

## Working Conventions

- Prefer fixing imports and call sites to match the current structure instead of adding compatibility packages.
- Keep public imports shallow and explicit; avoid implicit cross-package magic where a direct import is clearer.
- When moving files, update tests to the new module paths instead of reintroducing deprecated package surfaces.
- Keep `../IsaacLab` treated as vendored code; do not modify it to fix this repo.
- Keep lint and test configuration scoped to this repository so vendored code is not collected.
- Before adding abstractions, check whether the code can instead reuse an existing embodiment-common or task-local module.
- Add focused tests for refactor-sensitive helpers such as path resolution, config wiring, and observation/reward logic.
- Do not overwrite unrelated user changes in the worktree.
