# AdsorbDiff: Adsorbate Placement via Conditional Denoising Diffusion [ICML 2024]

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/AdeeshKolluru/AdsorbDiff/blob/main/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2305.01140-B31B1B.svg)](https://arxiv.org/abs/2405.03962)

This is the official code repository for the paper [AdsorbDiff: Adsorbate Placement via Conditional Denoising Diffusion](https://arxiv.org/abs/2405.03962), accepted at *International Conference on Machine Learning, 2024*.

If you have any questions, concerns, or feature requests, please feel free to email [me](mailto:kolluru.adeesh@gmail.com).

<img width="800" alt="adsorbdiff_mainfig" src="https://github.com/AdeeshKolluru/AdsorbDiff/assets/43401571/dfb1d2f4-9e56-4333-ae95-a8062e24af7a">


## Installation 

```
conda create -n adsorbdiff python=3.10
conda activate adsorbdiff
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

git clone https://github.com/AdeeshKolluru/AdsorbDiff.git
cd AdsorbDiff
pip install -r requirements.txt
pip install -e .

```

## Datasets

### Creating lmdbs for training a diffusion model -
- Download the ASE trajectories of OC20-Dense dataset [(link)](https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_trajectories.tar.gz) as well as [mappings](https://dl.fbaipublicfiles.com/opencatalystproject/data/adsorbml/oc20_dense_mappings.tar.gz).
- Download OC20-Dense subset of data in lmdb format with 244 adsorbate surface combinations which is utilized for Open Catalyst Challenge 2023 - [Link](https://dl.fbaipublicfiles.com/opencatalystproject/data/neurips_2023/oc20dense_is2re_train_v2.tar.gz)
- Find unique ids (sids) by iterating across the lmdb above and store it using [this](https://github.com/AdeeshKolluru/AdsorbDiff/blob/main/scripts/create_unique_train_system_id.py) script.
- We can then create the train and val ID lmdbs using [this](https://github.com/AdeeshKolluru/AdsorbDiff/blob/main/scripts/preprocess_train_lmdb_subsplits.py) script.
- For generating lmdbs for conditional training, [this](https://github.com/AdeeshKolluru/AdsorbDiff/blob/main/scripts/preprocess_train_all_lmdb.py) script can be used.

Or you can directly download the lmdbs for conditional diffusion training [here](https://zenodo.org/records/11152248/files/train_conditional_lmdb.tar.gz).

### Creating lmdbs for sampling -

- Val OOD ASE trajectories - [download](https://dl.fbaipublicfiles.com/opencatalystproject/data/neurips_2023/oc20dense_is2re_val_ase.tar.gz).
- For generating val OOD lmdbs from val OOD trajs, [this](https://github.com/AdeeshKolluru/AdsorbDiff/blob/main/scripts/preprocess_lmdb.py) script can be used.
  
Val OOD subset lmdb - [download](https://zenodo.org/records/11152248/files/valood50_R1I0.1.tar.gz)

- To generate lmdbs for AdsorbML baselines, [this](https://github.com/AdeeshKolluru/AdsorbDiff/blob/main/scripts/preprocess_val_relax_lmdb.py) script can be used.

Val ID lmdb - [download](https://zenodo.org/records/11152248/files/adsorbdiff_valID_lmdb.tar.gz)

### Pretraining on OC20 -
- OC20 IS2RE lmdbs - [download](https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz).

## Checkpoints

### Diffusion

PaiNN - [download](https://zenodo.org/records/11152248/files/PT_zeroshot_painn.pt) - _recommended due to faster inference_

EquiformerV2 - [download](https://zenodo.org/records/11152248/files/PT_fewshot_eqv2_cond.pt)

### MLFF optimization

All pre-trained OCP checkpoints can be downloaded [here](https://github.com/FAIR-Chem/fairchem/blob/74624e690a62c525f40fbff83df6fd45a0d14ab8/src/fairchem/core/models/pretrained_models.yml) for MLFF optimization.

## Training, sampling and relaxation
The bash script to generate commands for training, sampling and relaxations is in ```run.py```. Different commands in this file can be used for different cases.

## Citation
Please consider citing our paper if you find it helpful. Thank you!
```
@misc{kolluru2024adsorbdiff,
      title={AdsorbDiff: Adsorbate Placement via Conditional Denoising Diffusion}, 
      author={Adeesh Kolluru and John R Kitchin},
      year={2024},
      eprint={2405.03962},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
This codebase was built on top of [ocp](https://github.com/Open-Catalyst-Project/ocp), [Open-Catalyst-Dataset](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset) repositories as well as adapting code from [DiffDock](https://github.com/gcorso/DiffDock) and [AdsorbML](https://github.com/Open-Catalyst-Project/AdsorbML). Please cite these works as well!
