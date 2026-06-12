# swarm-lab

`swarm-lab` is the thesis workspace for multi-agent reinforcement learning on Isaac Sim and Isaac Lab.

The repository is split into two layers:

- `../cpsquare-lab` provides reusable robot embodiments and shared multirotor utilities.
- `swarm-lab` provides executable task packages, training scripts, evaluation tooling, and thesis documentation.

## Common Commands

```bash
uv run pytest test/
uv run ruff check .
uv run pyright
uv run --group docs mkdocs build --strict
```

## Documentation

Run the developer documentation locally with MkDocs:

```bash
uv run --group docs mkdocs serve
```

Build the static documentation site into `site/`:

```bash
uv run --group docs mkdocs build --strict
```

The generated site is a build artifact and should not be committed.
