from config import Config
from src.models.train_model import SimpleCNN, MyEfficientNet, PreTrainedEfficientNet, HuggingfaceResNet

import os
import torch

config = Config()


model_selector = {
    'CNN': SimpleCNN().to(config.device),
    'EN': MyEfficientNet().to(config.device),
    'EN_PT': PreTrainedEfficientNet().to(config.device),
    'RN_PT': HuggingfaceResNet().to(config.device),
}

params_files = {
    'CNN': os.path.join(config.model_dir, 'params/CNN_Intel_params_n20.pt'),
    'EN': os.path.join(config.model_dir, 'params/EN_Intel_params_n20.pt'),
    'EN_PT': os.path.join(config.model_dir, 'params/EN_PT_Intel_params_n20.pt'),
    'RN_PT': os.path.join(config.model_dir, 'params/RN_PT_Intel_params_n20.pt'),
}

train_results_files = {
    'CNN': os.path.join(config.model_dir, 'train_results/CNN_Intel_results_n20.json'),
    'EN': os.path.join(config.model_dir, 'train_results/EN_Intel_results_n20.json'),
    'EN_PT': os.path.join(config.model_dir, 'train_results/EN_PT_Intel_results_n20.json'),
    'RN_PT': os.path.join(config.model_dir, 'train_results/RN_PT_Intel_results_n20.json'),
}

preds_files = {
    'CNN': os.path.join(config.model_dir, 'preds/CNN_Intel_predictions.json'),
    'EN': os.path.join(config.model_dir, 'preds/EN_Intel_predictions.json'),
    'EN_PT': os.path.join(config.model_dir, 'preds/EN_PT_Intel_predictions.json'),
    'RN_PT': os.path.join(config.model_dir, 'preds/RN_PT_Intel_predictions.json'),
}

evals_files = {
    'CNN': os.path.join(config.model_dir, 'evals/CNN_Intel_evaluation.json'),
    'EN': os.path.join(config.model_dir, 'evals/EN_Intel_evaluation.json'),
    'EN_PT': os.path.join(config.model_dir, 'evals/EN_PT_Intel_evaluation.json'),
    'RN_PT': os.path.join(config.model_dir, 'evals/RN_PT_Intel_evaluation.json'),
}