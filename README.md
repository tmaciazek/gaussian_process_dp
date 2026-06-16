# Diﬀerential Privacy of Gaussian Process Posterior Sampling
This repository is the official implementation of [Diﬀerential Privacy of Gaussian Process Posterior Sampling](https://arxiv.org/).

Diﬀerential Privacy of Gaussian Process Posterior Sampling. Probing DP guarantees by MIA and privacy-utility tradeoff via excursion sets.

## LiRA Membership Inference Attack

To recreate the LiRA attack results for $r=0.5$ and $\sigma=0.5$ run the following command:
```
python lira_fhat_vhat_latent_gmm_exp1d_logv.py \
  --n 10 \
  --ell 1 \
  --r 0.1 \
  --sigma 0.5 \
  --n-posterior-draws 10 \
  --n-shadow 10000 \
  --n-eval 10000 \
  --seed 0 \
  --save-dir lira_exp1D_results
```
