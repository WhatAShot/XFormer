# XTFormer

## Environment Setup

First, create a new conda environment and install the required packages.

```shell
conda create -n xtformer python=3.10
conda activate xtformer
```

Then, install the required packages.

```shell
pip install -r requirements.txt
```

## Data Preparation

Download the data from the Google Drive

**NOTE: The full dataset will be released as the paper is accepted.**

And extract the data to the `data` directory.

Before running the code, you need to set the `PROJ` path in the `lib/env.py` file in line 8.

```python
# NOTE: SET YOUR PROJECT ROOT HERE
PROJ = Path("YOUR DIR").absolute().resolve()
```

## Training

To train the model, run the following command.

```shell
python train.py
```

In this script, you can both pretrain and fine-tune the model. If you want to fine-tune the model, you need to set the pretrained model path in the `pretrain.py` file in line 927.

```python
# NOTE: SET THE PATH TO THE PRE-TRAINED MODEL
params_path = 'YOUR PRETRAINED MODEL PATH'
```