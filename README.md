# Diﬀerential Privacy of Gaussian Process Posterior Sampling
This repository is the official implementation of [Diﬀerential Privacy of Gaussian Process Posterior Sampling](https://arxiv.org/).

Diﬀerential Privacy of Gaussian Process Posterior Sampling. Probing DP guarantees by MIA and privacy-utility tradeoff via excursion sets.

This repository contains the main scripts used for the experiments in the paper. We are currently cleaning and documenting the remaining plotting and 2D-experiment scripts. They will be added in a subsequent update.

## LiRA Membership Inference Attack

Example: to recreate the LiRA attack results for $r=0.1$, $\sigma=0.5$ and $L=3$ posterior draws run the following command:
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
    --n-train 100 \ 
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

## 2D Excursion Set Experiments
Example: to recreate the 2D excurion set results for $M_\xi=0.5$ just set `--M-xi 0.5` below. 
```
python run_2d_excursion_gp_private_sigmoid_smoothed.py --M-xi 0.5
```

## London Property Sales Experiment
First download the [Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) from 2018 and 2017. Then, download the [postcode lookup table](https://geoportal.statistics.gov.uk/datasets/6fff67d204fd4f339591ed667a6e3642/about). If the property sales data is `pp-2017.csv` and the postcode lookup table is in the file `NSPL.csv` then to fit the GP hyperparameters run:
```
python fit_hexbin_gp_excursion_london_private.py \
  --ppd pp-2017.csv \
  --postcode-lookup NSPL.csv \
  --threshold 13.0 \
  --gridsize 40 \
  --mincnt 1 \
  --epsilon0 10 \
  --M-Y 1.0 \
  --L 1 \
  --out-figure london_hexbin_gp_public_excursion.png \
  --out-private-figure london_hexbin_gp_private_excursion.png \
  --out-summary london_hexbin_gp_private_summary.json \
  --no-show
```
Then, apply those hyperparameters to find the excursion set for 2018 data as follows:
```
python fit_hexbin_gp_excursion_london_private_json.py \
  --ppd pp-2018.csv \
  --postcode-lookup NSPL.csv \
  --threshold 13.0 \
  --response-scale 1.0 \
  --gridsize 40 \
  --mincnt 3 \
  --n-sample-paths 3 \
  --path-grid-size 60 \
  --path-seed 123
```
