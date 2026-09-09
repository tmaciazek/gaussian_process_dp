#!/usr/bin/env bash
set -euo pipefail

# Generate the compact LiRA ROC files consumed by lira_plots_updated.py.
# The Python attack script always writes lira_exp1D_results/lira_*.npy;
# --no-save only suppresses the larger diagnostic .npz/histogram outputs.
#
# Usage:
#   ./mia_submit.sh single
#   ./mia_submit.sh composition
#
# Optional environment overrides:
#   PYTHON=python3 N_SHADOW=10000 N_EVAL=10000 ./mia_submit.sh single

MODE="${1:-}"
PYTHON="${PYTHON:-python}"
ATTACK_SCRIPT="${ATTACK_SCRIPT:-lira_fhat_vhat_latent_gmm_exp1d_logv.py}"
N_SHADOW="${N_SHADOW:-10000}"
N_EVAL="${N_EVAL:-10000}"

run_one() {
    local r="$1"
    local sigma="$2"
    local L="$3"

    echo "r=${r} sigma=${sigma} L=${L}"
    local pids=()
    for seed in {0..9}; do
        "$PYTHON" "$ATTACK_SCRIPT" \
            --n 10 \
            --x0 0.5 \
            --ell 1 \
            --r "$r" \
            --sigma "$sigma" \
            --m-eps 0 \
            --n-posterior-draws "$L" \
            --n-shadow "$N_SHADOW" \
            --n-eval "$N_EVAL" \
            --seed "$seed" \
            --no-save &
        pids+=("$!")
    done

    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if (( failed )); then
        echo "At least one seed failed for r=${r}, sigma=${sigma}, L=${L}." >&2
        return 1
    fi
}

case "$MODE" in
    single)
        # Matches plot_lira_vs_epsilon(): 30 log-spaced r values, L=1,
        # sigma in {0.5, 1, 2.5, infinity}.  sigma=inf is simulated by the
        # covariance-only scaled limit implemented in the attack script.
        R_VALUES=$(
            "$PYTHON" - <<'PY'
import numpy as np
for r in np.logspace(np.log10(0.065), np.log10(4.0), 30):
    print(f"{r:.17g}")
PY
        )
        SIGMAS=(0.5 1 2.5 inf)

        for sigma in "${SIGMAS[@]}"; do
            for r in $R_VALUES; do
                run_one "$r" "$sigma" 1
            done
        done
        ;;

    composition)
        # Matches plot_lira_and_epsilon_vs_L() in the attached plotting script.
        # If that plot is changed to another sigma, change COMPOSITION_SIGMA here too.
        COMPOSITION_R=1
        COMPOSITION_SIGMA=2
        L_VALUES=(1 2 5 10 22 46 100 215 464 1000)

        for L in "${L_VALUES[@]}"; do
            run_one "$COMPOSITION_R" "$COMPOSITION_SIGMA" "$L"
        done
        ;;

    *)
        echo "Usage: $0 {single|composition}" >&2
        exit 2
        ;;
esac
