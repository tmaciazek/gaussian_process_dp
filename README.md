# Diﬀerential Privacy of Gaussian Process Posterior Sampling
This repository is the official implementation of [Diﬀerential Privacy of Gaussian Process Posterior Sampling](https://arxiv.org/).

Diﬀerential Privacy of Gaussian Process Posterior Sampling. Probing DP guarantees by MIA and privacy-utility tradeoff via excursion sets.

## LiRA Membership Inference Attack

Example: to recreate the LiRA attack results for $r=0.5$, $\sigma=0.5$ and $L=3$ posterior draws run the following command:
```
python lira_fhat_vhat_latent_gmm_exp1d_logv.py \
  --n 10 \
  --ell 1 \
  --r 0.1 \
  --sigma 0.5 \
  --n-posterior-draws 3 \
  --n-shadow 10000 \
  --n-eval 10000 \
  --seed 0 \
  --save-dir lira_exp1D_results
```

## 1D Excursion Set Experiments
Example: to recreate the 1D excurion set results for $M_\xi=0.5$ just set `--m-eps 0.5` below. 
```
python nonprivate_gridsearch_exp_excursion_set_bce.py\
    --n-train 10 \ 
    --n-target-grid 800 \
    --ell-true 1. \
    --ell-model 0.08,0.13,0.2,0.25,0.35,0.5,0.6,0.75,1 \
    --r-values 0.05,0.1,0.2,0.5,1,2,5 \
    --sigma-values 0,0.1,0.5,1,2,5 \
    --n-trials 100 \
    --n-posterior-draws 50 \
    --m-eps 0.5 \
    --threshold 0.0 \
    --epsilon-threshold 3 \
    --seed 6 \
    --target-reject \
    --target-reject-volume-min 0.1 \
    --target-reject-volume-max 0.9 \
    --target-reject-max-components 13 \
    --target-reject-min-mean-component-width 0.05
```
