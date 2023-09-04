from config import Config

import numpy as np
import torch

config = Config()


def get_all_predictions(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend([int(x) for x in preds.cpu().numpy()])
            all_labels.extend([int(x) for x in labels.cpu().numpy()])

    return np.array(all_preds), np.array(all_labels)
