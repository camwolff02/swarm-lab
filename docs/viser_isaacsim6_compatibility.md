# Viser / IsaacSim 6 Compatibility Note

Date observed: 2026-04-27.

The Viser frontend is currently not a supported launch path in this workspace.

## Observed stack

- IsaacSim: `6.0.0`
- `isaacsim-kernel`: `6.0.0.0`
- `newton`: `1.0.0`
- Compatible resolver fallback: `viser==0.2.11`
- IsaacSim-required websocket version: `websockets==12.0`

## Failure modes

Requesting the IsaacLab-advertised Viser dependency (`viser>=1.0.16`) makes the
workspace unsatisfiable:

- `isaacsim-kernel==6.0.0.0` pins `websockets==12.0`.
- `viser>=1.0.16` requires `websockets>=13.1`.

Keeping the older Viser line (`viser==0.2.11`) resolves with IsaacSim 6, but the
current `newton.viewer.ViewerViser` expects a newer Viser scene API:

```text
AttributeError: 'SceneApi' object has no attribute 'configure_environment_map'
```

In `viser==0.2.11`, the equivalent method is named `set_environment_map`.

## Current decision

Do not use `--viz viser` for now. Use Kit, Newton, Rerun, or no frontend while
the quad swarm environment work continues.

The project dependency remains capped to `viser>=0.2.11,<1.0.16` so `uv run`
stays resolvable with IsaacSim 6. This cap is not sufficient to make the Viser
frontend work with the current Newton viewer; it only prevents uv from selecting
the known unsatisfiable Viser 1.x dependency set.

## Follow-up options

- Find a Newton version whose `ViewerViser` still targets Viser 0.2.x.
- Use a patched/local Viser build that both supports `configure_environment_map`
  and allows IsaacSim's `websockets==12.0`.
- Upgrade to an IsaacSim/IsaacLab stack where the IsaacSim websocket pin and
  IsaacLab's Viser requirement are mutually compatible.
