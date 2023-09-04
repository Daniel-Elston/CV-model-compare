from config import Config
from src.models.predict_model import get_all_predictions

import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

config = Config()


class ModelEvaluator():
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def compute_performance_metrics(self):
        preds, labels = get_all_predictions(self.model, self.loader)

        f1 = f1_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        accuracy = accuracy_score(labels, preds)
        
        performance_metrics = {
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy
        }
        return performance_metrics
    
    def plot_confusion_matrix(self):
        preds, labels = get_all_predictions(self.model, self.loader)
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def class_specific_accuracy(self):
        preds, labels = get_all_predictions(self.model, self.loader)
        cm = confusion_matrix(labels, preds)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        class_accuracy_dict = {f"Class {i}": acc for i, acc in enumerate(class_accuracy)}
        return class_accuracy_dict

    def computational_and_efficiency_metrics(self):        
        # Measure inference time for a single pass
        start_time = time.time()
        get_all_predictions(self.model, self.loader)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        # Model size (parameters + data)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        computational_metrics = {
            'Inference Time': inference_time,
            'Total Parameters': total_params
        }
        return computational_metrics
        
    def performance_vs_complexity(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        preds, labels = get_all_predictions(self.model, self.loader)
        accuracy = accuracy_score(labels, preds)
        
        complexity_metrics = {
            'Total Parameters': total_params,
            'Accuracy': accuracy
        }
        return complexity_metrics