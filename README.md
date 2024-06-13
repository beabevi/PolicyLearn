# Efficient Subgraph GNNs by Learning Effective Selection Policies

This repository contains the official code of the paper
**[Efficient Subgraph GNNs by Learning Effective Selection Policies](https://arxiv.org/abs/2310.20082) (ICLR 2024)**.


<p align="center">
<img src=./policy-learn.png>
</p>

## Reproduce results

To perform hyperparameter tuning, make use of `wandb`:

1. In the `yaml-files` folder, choose the `yaml` file corresponding to the dataset of interest, say `<config-name>`.
    This file contains the hyperparameters grid.

2. Run
    ```bash
    wandb sweep yaml-files/<config-name>
    ````
    to obtain a sweep id `<sweep-id>`

3. Run the hyperparameter tuning with
    ```bash
    wandb agent <sweep-id>
    ```
    You can run the above command multiple times on each machine you would like to contribute to the grid-search

4. Open your project in your wandb account on the browser to see the results:

    * Compute mean and std of `best val`, `test metric @ best val` by grouping over all hyperparameters and averaging over the different seeds.
    Then, take the results corresponding to the configuration obtaining the best validation metric.



## Credits

For attribution in academic contexts, please cite

```
@inproceedings{bevilacqua2024efficient,
title={Efficient {S}ubgraph {GNN}s by Learning Effective Selection Policies},
author={Beatrice Bevilacqua and Moshe Eliasof and Eli Meirom and Bruno Ribeiro and Haggai Maron},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
}
```