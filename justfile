set shell := ["bash", "-uc"]

formation_log_root := "logs/skrl/formation_swarm"
formation_task := "Isaac-Formation-Swarm-MAPPO-v0"
formation_algorithm := "MAPPO"

_latest-formation-checkpoint:
    @checkpoint="$(find {{formation_log_root}} -path '*/checkpoints/agent_*.pt' -type f -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"; \
    if [[ -z "$checkpoint" ]]; then \
        echo "No formation checkpoint found under {{formation_log_root}}" >&2; \
        exit 1; \
    fi; \
    echo "$checkpoint"

formation-train-stage stage checkpoint="" max_iterations="":
    #!/usr/bin/env bash
    set -euo pipefail
    cmd=(
        uv run scripts/skrl/train.py
        --task "{{formation_task}}"
        --algorithm "{{formation_algorithm}}"
        env.curriculum_stage="{{stage}}"
    )
    if [[ -n "{{checkpoint}}" ]]; then
        cmd+=(--checkpoint "{{checkpoint}}" --reset_optimizer_on_resume)
    fi
    if [[ -n "{{max_iterations}}" ]]; then
        cmd+=(--max_iterations "{{max_iterations}}")
    fi
    printf '[INFO] Running stage %s' "{{stage}}"
    if [[ -n "{{checkpoint}}" ]]; then
        printf ' from %s' "{{checkpoint}}"
    fi
    printf '\n'
    "${cmd[@]}"

formation-play stage="3" checkpoint="":
    #!/usr/bin/env bash
    set -euo pipefail
    chosen_checkpoint="{{checkpoint}}"
    if [[ -z "$chosen_checkpoint" ]]; then
        chosen_checkpoint="$(find "{{formation_log_root}}" -path '*/checkpoints/agent_*.pt' -type f -printf '%T@ %p\n' \
            | sort -nr \
            | awk 'NR==1 {print $2}')"
    fi
    if [[ -z "$chosen_checkpoint" ]]; then
        echo "No formation checkpoint found under {{formation_log_root}}" >&2
        exit 1
    fi
    printf '[INFO] Playing curriculum stage %s from %s\n' "{{stage}}" "$chosen_checkpoint"
    uv run scripts/skrl/play.py \
        --task "{{formation_task}}" \
        --algorithm "{{formation_algorithm}}" \
        --checkpoint "$chosen_checkpoint" \
        env.curriculum_stage="{{stage}}"

formation-curriculum max_iterations="":
    #!/usr/bin/env bash
    set -euo pipefail
    latest_checkpoint() {
        find "{{formation_log_root}}" -path '*/checkpoints/agent_*.pt' -type f -printf '%T@ %p\n' \
            | sort -nr \
            | awk 'NR==1 {print $2}'
    }
    run_stage() {
        local stage="$1"
        local checkpoint="${2:-}"
        local cmd=(
            uv run scripts/skrl/train.py
            --task "{{formation_task}}"
            --algorithm "{{formation_algorithm}}"
            env.curriculum_stage="$stage"
        )
        if [[ -n "$checkpoint" ]]; then
            cmd+=(--checkpoint "$checkpoint" --reset_optimizer_on_resume)
        fi
        if [[ -n "{{max_iterations}}" ]]; then
            cmd+=(--max_iterations "{{max_iterations}}")
        fi
        printf '[INFO] Starting curriculum stage %s' "$stage"
        if [[ -n "$checkpoint" ]]; then
            printf ' from %s' "$checkpoint"
        fi
        printf '\n'
        "${cmd[@]}"
    }

    run_stage 1
    stage1_checkpoint="$(latest_checkpoint)"
    if [[ -z "$stage1_checkpoint" ]]; then
        echo "Stage 1 finished but no checkpoint was found." >&2
        exit 1
    fi

    run_stage 2 "$stage1_checkpoint"
    stage2_checkpoint="$(latest_checkpoint)"
    if [[ -z "$stage2_checkpoint" || "$stage2_checkpoint" == "$stage1_checkpoint" ]]; then
        echo "Stage 2 finished but no new checkpoint was found." >&2
        exit 1
    fi

    run_stage 3 "$stage2_checkpoint"
    stage3_checkpoint="$(latest_checkpoint)"
    if [[ -z "$stage3_checkpoint" || "$stage3_checkpoint" == "$stage2_checkpoint" ]]; then
        echo "Stage 3 finished but no new checkpoint was found." >&2
        exit 1
    fi
    printf '[INFO] Curriculum complete. Final checkpoint: %s\n' "$stage3_checkpoint"
