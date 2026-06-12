# ROS2 Bridge

The shared ROS2 bridge now lives in `cpsquare_lab`:

- `cpsquare_lab/ros2/common.py`
- `cpsquare_lab/ros2/agent_bridge.py`
- `cpsquare_lab/ros2/environment_bridge.py`

The local repo only keeps two thin launchers in `scripts/ros2`:

- `scripts/ros2/run_env.py`
- `scripts/ros2/run_agent.py`

`run_env.py` still handles Isaac Lab app startup. The ROS2 transport, schema generation, topic resolution, flattening, logging, and agent loading now live in `cpsquare_lab`.

## Requirements

The bridge uses `zenoh_ros2_sdk`. For cross-process communication you should have a Zenoh router running, typically on `127.0.0.1:7447`.

## Topics And Messages

Topics are derived from the task name by default:

- schema: `/bridge/<task>/schema`
- observations: `/bridge/<task>/observations`
- actions: `/bridge/<task>/actions`
- status: `/bridge/<task>/status`

Message types:

- schema: `std_msgs/msg/String`
- status: `std_msgs/msg/String`
- observations: `example_interfaces/msg/Float32MultiArray`
- actions: `example_interfaces/msg/Float32MultiArray`

Schema and status are JSON strings. Observations and actions are flattened float32 arrays. The schema describes the original nested observation and action tree layout.

## Running The Environment

Show CLI help:

```bash
uv run scripts/ros2/run_env.py --help
```

Run the environment with a viewer:

```bash
uv run scripts/ros2/run_env.py \
  --task Isaac-SimpleFlight-Classical-Crazyflie-v0 \
  --num_envs 1 \
  --viz kit
```

Run headless:

```bash
uv run scripts/ros2/run_env.py \
  --task Isaac-SimpleFlight-Classical-Crazyflie-v0 \
  --num_envs 1
```

Useful flags:

- `--publish_every`: publish observations every N environment steps
- `--max_steps`: stop after N environment steps
- `--wait_for_first_action`: block until the first action arrives
- `--action_timeout`: maximum allowed age of the latest action
- `--timeout_behavior zero|hold-last`: how to handle stale actions
- `--key_prefix`: override the `/bridge/<task>` prefix

Logs are written under:

```text
logs/ros2/<task>/<timestamp>_ros2/
```

Each run directory includes:

- `env.yaml`
- `args.json`
- `schema.json`
- `events.jsonl`
- `summary.json`

## Running An Agent

Show CLI help:

```bash
uv run scripts/ros2/run_agent.py --help
```

Run the default bridge agent registered for the task:

```bash
uv run scripts/ros2/run_agent.py \
  --task Isaac-SimpleFlight-Classical-Crazyflie-v0
```

Run a named task agent:

```bash
uv run scripts/ros2/run_agent.py \
  --task Isaac-SimpleFlight-Classical-Crazyflie-v0 \
  --agent hover
```

Task-specific agents live under `environments/environments/tasks/<task>/agents/` and are registered by the owning task package.

## Schema Format

The schema JSON contains:

- `task`
- `num_envs`
- `step_dt`
- `topics`
- `observation_spec`
- `action_spec`

Example:

```json
{
  "root_type": "dict",
  "total_size": 58,
  "leaves": [
    { "path": ["controller"], "shape": [1, 16], "size": 16 },
    { "path": ["policy"], "shape": [1, 42], "size": 42 }
  ]
}
```

Leaves are flattened in schema order. Agents use the same spec to rebuild the structured observation and action trees.

## Testing

Run the unit tests with:

```bash
uv run python -m pytest test
```

Coverage currently includes:

- flatten/unflatten and schema helpers
- `AgentBridge` message handling and action publishing
- `EnvironmentBridge` publish/receive behavior
- `run_agent.py` launcher wiring
- task launcher wiring
