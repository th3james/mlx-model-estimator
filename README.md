# MLX-LM Model Estimator

Lots of quantised model variants are published to [mlx-community on Hugging Face](https://huggingface.co/mlx-community), but which is best for your Mac?

This python snippet grabs a selection of models and runs [Arc-Easy](https://huggingface.co/datasets/allenai/ai2_arc) to provide an estimate of how much the model has been degraded by quantisation. It also provides very rough timing numbers.

## Install
Tested with Python 3.12

```sh
pip install -r requirements.txt
```

## Configure

Set the HF models you want to test and configure the number of questions by editing the constants in `evaluate_models.py`.

## Run

```sh
python evaluate_models.py
```
