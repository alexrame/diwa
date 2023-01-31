# Diverse Weight Averaging for Out-of-Distribution Generalization, NeurIPS 2022

Official PyTorch implementation of DiWA | [paper](https://arxiv.org/abs/2205.09739), [openreview](https://openreview.net/forum?id=tq_J_MqB3UB)

[Alexandre Ramé](https://alexrame.github.io/), [Matthieu Kirchmeyer](https://scholar.google.com/citations?user=oJkKtrkAAAAJ&hl=en), [Thibaud Rahier](https://scholar.google.nl/citations?user=IFOAOTMAAAAJ&hl=pl), [Alain Rakotomamonjy](http://asi.insa-rouen.fr/enseignants/~arakoto/), [Patrick Gallinari](http://www-connex.lip6.fr/~gallinar/gallinari/pmwiki.php), [Matthieu Cord](http://webia.lip6.fr/~cord/)

## TL;DR

To improve out-of-distribution generalization, we average diverse weights obtained from different training runs; this strategy is motivated by an extension of the bias-variance theory to weight averaging and is state-of-the-art on DomainBed.

![diwa](https://user-images.githubusercontent.com/15007187/215759479-cb060691-cba4-4ef4-832d-9269e61c951a.png)

## Abstract

Standard neural networks struggle to generalize under distribution shifts. For out-of-distribution generalization in computer vision, the best current approach averages the weights along a training run. In this paper, we propose Diverse Weight Averaging (DiWA) that makes a simple change to this strategy: DiWA averages the weights obtained from several independent training runs rather than from a single run. Perhaps surprisingly, averaging these weights performs well under soft constraints despite the network's nonlinearities. The main motivation behind DiWA is to increase the functional diversity across averaged models. Indeed, models obtained from different runs are more diverse than those collected along a single run thanks to differences in hyperparameters and training procedures. We motivate the need for diversity by a new bias-variance-covariance-locality decomposition of the expected error, exploiting similarities between DiWA and standard functional ensembling. Moreover, this decomposition highlights that DiWA succeeds when the variance term dominates, which we show happens when the marginal distribution changes at test time. Experimentally, DiWA consistently improves the state of the art on the competitive DomainBed benchmark without inference overhead.

## DomainBed

Our code is adapted from the open-source [DomainBed github](https://github.com/facebookresearch/DomainBed/), which is a PyTorch benchmark including datasets and algorithms for Out-of-Distribution generalization. It was introduced in [In Search of Lost Domain Generalization, ICLR 2021](https://openreview.net/forum?id=lQdXeXDoWtI).

In addition to the newly-added `domainbed/scripts/diwa.py` and `domainbed/algorithms_inference.py` files, we made only few modifications to this codebase, all preceded by `## DiWA ##`.

* in `domainbed/hparams_registry.py`, to define our mild hyperparameter ranges.
* in `domainbed/train.py`, to handle the shared initialization and save the weights of the epoch with the highest validation accuracy.
* in `domainbed/algorithms.py`, to handle the shared initialization, the linear probing approach and implement the MA baseline.
* in `domainbed/datasets.py`, to define the checkpoint frequency.
* in `domainbed/scripts/sweep.py`, to be able to force the test env.
* in `domainbed/lib/misc.py`, to include some tools.

Then you should be able to reproduce our main experiment (Table 1) on the DomainBed benchmark.

### Requirements

* python == 3.7.10
* torch == 1.8.1
* torchvision == 0.9.1
* numpy == 1.20.2

### Datasets

We ran DiWA on the following [datasets](domainbed/datasets.py):

* VLCS ([Fang et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* OfficeHome ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
* A TerraIncognita ([Beery et al., 2018](https://arxiv.org/abs/1807.04975)) subset
* DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))
* Colored MNIST ([Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))

You can download the datasets with following command:

```sh
python3 -m domainbed.scripts.download --data_dir=/my/data/dir
```

## DiWA Procedure Details

Our training procedure is in three stages.

### Set the initialization

First, we need to fix the initialization.

```sh
python3 -m domainbed.scripts.train\
       --data_dir=/my/data/dir/\
       --algorithm ERM\
       --dataset OfficeHome\
       --test_env ${test_env}\
       --init_step\
       --path_for_init ${path_for_init}\
       --steps ${steps}\
```

In the paper, we proposed $2$ initialization procedures:

* random initialization, set `steps` to `-1`: there will be no training.
* [Linear Probing, ICLR2022](https://openreview.net/forum?id=UYneFzXSJWh), set `steps` to `0`: only the classifier will be trained.

The initialization is then saved at `${path_for_init}`, to be used in the subsequent sweep.

### Launch ERM training

Second, we launch several ERM runs following the hyperparameter distributions from [here](domainbed/hparams_registry.py), as defined in Table 5 from Appendix F.1.
To do so, we leverage the native `sweep` script from DomainBed.

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/data/dir/\
       --output_dir=/my/sweep/output/path\
       --command_launcher multi_gpu\
       --datasets OfficeHome\
       --test_env ${test_env}\
       --path_for_init ${path_for_init}\
       --algorithms ERM\
       --n_hparams 20\
       --n_trials 3
```

### Average the diverse weights

Finally, we average the weights obtained from this grid search.

```sh
python -m domainbed.scripts.diwa\
       --data_dir=/my/data/dir/\
       --output_dir=/my/sweep/output/path\
       --dataset OfficeHome\
       --test_env ${test_env}\
       --weight_selection ${weight_selection}
       --trial_seed ${trial_seed}
```

In the paper, we proposed $3$ different procedures:

* DiWA-restricted, set `weight_selection` to `restricted` and `trial_seed` to an integer between `0` and `2`.
* DiWA-uniform, set `weight_selection` to `uniform` and `trial_seed` to an integer between `0` and `2`.
* DiWA$^\dagger$-uniform, set `weight_selection` to `uniform` and `trial_seed` to `-1`.

## Weight averaging from a single run

You can reproduce the [Moving Average (MA)](https://arxiv.org/abs/2110.10832) baseline by replacing ERM by MA as the algorithm argument.

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/data/dir/\
       --output_dir=/my/sweep/output/path\
       --command_launcher multi_gpu\
       --datasets OfficeHome\
       --test_env ${test_env}\
       --algorithms MA\
       --n_hparams 20\
       --n_trials 3
```

Then to view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results --input_dir=/my/sweep/output/path
````

# Results

DiWA sets a new state of the art on DomainBed.

| Algorithm        | Weight selection | Init   | PACS | VLCS | OfficeHome | TerraInc | DomainNet | Avg  |
|---|---|---|---|---|---|---|---|---|
| ERM              | N/A              | Random | 85.5 | 77.5 | 66.5       | 46.1     | 40.9      | 63.3 |
| Coral            | N/A              | Random | 86.2 | 78.8 | 68.7       | 47.6     | 41.5      | 64.6 |
| SWAD             | Overfit-aware    | Random | 88.1 | 79.1 | 70.6       | 50.0     | 46.5      | 66.9 |
| MA               | Uniform          | Random | 87.5 | 78.2 | 70.6       | 50.3     | 46.0      | 66.5 |
|---|---|---|---|---|---|---|---|---|
| ERM              | N/A              | Random | 85.5 | 77.6 | 67.4       | 48.3     | 44.1      | 64.6 |
| DiWA             | Restricted       | Random | 87.9 | 79.2 | 70.5       | 50.5     | 46.7      | 67.0 |
| DiWA             | Uniform          | Random | 88.8 | 79.1 | 71.0       | 48.9     | 46.1      | 66.8 |
| DiWA$^{\dagger}$ | Uniform          | Random | 89.0 | 79.4 | 71.6       | 49.0     | 46.3      | 67.1 |
|---|---|---|---|---|---|---|---|---|
| ERM              | N/A              | LP     | 85.9 | 78.1 | 69.4       | 50.4     | 44.3      | 65.6 |
| DiWA             | Restricted       | LP     | 88.0 | 78.5 | 71.5       | 51.6     | 47.7      | 67.5 |
| DiWA             | Uniform          | LP     | 88.7 | 78.4 | 72.1       | 51.4     | 47.4      | 67.6 |
| DiWA$^{\dagger}$ | Uniform          | LP     | 89.0 | 78.6 | 72.8       | 51.9     | 47.7      | 68.0 |

# Citation

If you find this code useful for your research, please consider citing our work:

```
@inproceedings{rame2022diwa,
  title   = {Diverse Weight Averaging for Out-of-Distribution Generalization},
  author  = {Rame, Alexandre and Kirchmeyer, Matthieu and Rahier, Thibaud and Rakotomamonjy, Alain and Gallinari, Patrick and Cord, Matthieu},
  year    = {2022},
  booktitle = {NeurIPS}
}
```

Correspondence to alexandre.rame at sorbonne-universite dot fr
