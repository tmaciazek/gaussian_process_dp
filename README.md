# Diﬀerential Privacy of Gaussian Process Posterior Sampling
This repository is the official implementation of [Diﬀerential Privacy of Gaussian Process Posterior Sampling](https://arxiv.org/).

Diﬀerential Privacy of Gaussian Process Posterior Sampling. Probing DP guarantees by MIA and privacy-utility tradeoff via excursion sets.

This repository contains the main scripts used for the experiments in the paper. We are currently cleaning and documenting the remaining plotting and 2D-experiment scripts. They will be added in a subsequent update.

## Plotting approximate-DP Bounds

To reproduce Figure 1 from the paper run:
```
python plot_dp_contours.py
```
and to reproduce Figure 5 from Appendix B.5 showing how many sample paths can be released at a given privacy budget run for example:
```
python plot_lmax_contours.py --epsilon-budget 10.0
```

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
To reproduce Figures 2 and 3 from the paper run the full workflow generating data and plots as follows:
```
chmod +x mia_submit.sh

./mia_submit.sh single
python lira_plots_updated.py --mode single

./mia_submit.sh composition
python lira_plots_updated.py --mode composition
```

To reproduce the histograms from Figure 7 run
```
python "plot_lira_shadow_histograms.py" \
  --lira-script lira_fhat_vhat_latent_gmm_exp1d_logv.py \
  --pairs "0.1,0.5;0.1,100;2,0.5" \
  --n 10 \
  --x0 0.5 \
  --ell 1 \
  --m-eps 0 \
  --L 3 \
  --n-shadow 10000 \
  --seed 0 \
  --dp-n 10 \
  --dp-delta 0.001 \
  --dp-grid-size 31 \
  --bins 55 \
  --n-cols 3 \
  --no-share-x \
  --out-dir mia_histograms \
  --basename lira_histograms \
  --formats pdf,png
```

## 1D Excursion Set Experiments
To recreate the 1D excurion set results for $M_\xi=0.5$ just set `--m-eps 0.5` below. 
```
python prior_average_gridsearch_exp_excursion_set_bce_L_release.py --n-target-grid 800 \
    --ell-true 1. \
    --ell-model 0.05,0.08,0.13,0.2,0.25,0.35,0.5,0.6,0.75 \
    --r-values 0.1,0.2,0.5,1,2,5,6,7 \
    --sigma-values 0,0.1,0.5,0.7,1 \
    --n-trials 100 \
    --n-posterior-draws 50 \
    --m-eps 0.5 \
    --threshold 0.0 \
    --epsilon-threshold 9 \
    --seed 6 \
    --target-reject \
    --target-reject-volume-min 0.1 \
    --target-reject-volume-max 0.9 \
    --target-reject-max-components 13 \
    --target-reject-min-mean-component-width 0.05 --epsilon-delta 0.001 --n-pairs 1000 --n-train 100
```

## 2D Excursion Set Experiments
To recreate the 2D excurion set results for $M_\xi=0.5$ just set `--M-xi 0.5` below. 
```
python run_2d_excursion_gp_private_sigmoid_smoothed.py --M-xi 0.5
```
Then, to reproduce the corresponding column of Table 4 in Appendix D run 
```
python summarise_fixed_2d_excursion_hyperparams.py --M-xi 0.5
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
