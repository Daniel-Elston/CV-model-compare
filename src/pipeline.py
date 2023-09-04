from config import Config
from src.data.make_dataset import RawDataHandler, DatasetHandler
from src.models.train_model import ModelFitter
from src.models.predict_model import get_all_predictions
from src.models.evaluate_model import ModelEvaluator

import os
import json

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

config = Config()


class DataPipeline:
    def __init__(self, raw_data_path, data_path, batch_size=64):
        self.raw_data_path = raw_data_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None
        self.pred_loader = None

    def process_raw_data(self):
        handle_raw_data = RawDataHandler(self.raw_data_path)
        df = handle_raw_data.fetch_raw_data()
        df = handle_raw_data.encode_labels()
        df.to_parquet(self.data_path)

    def load_data(self):
        data_handler = DatasetHandler(self.data_path)
        _, _, _, _ = data_handler.get_datasets()
        self.train_loader, self.test_loader, self.pred_loader = data_handler.get_loaders(batch_size=self.batch_size)

    def run(self):
        self.process_raw_data()
        self.load_data()



class ModelTrainingPipeline:
    def __init__(self, model_selector, train_loader, test_loader, epochs=20, lr=0.001):
        self.model_selector = model_selector
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_dir = config.model_dir
        self.epochs = epochs
        self.lr = lr
        
    def save_train_results(self, model_name, model_instance, history_store):
        accuracy = round(history_store[-1].get('val_acc', 0), 3)
        history_path = os.path.join(self.model_dir, f'train_results/{model_name}_Intel_results_n{self.epochs}.json')
        model_path = os.path.join(self.model_dir, f'params/{model_name}_Intel_params_n{self.epochs}.pt')
        
        # Save the training results
        with open(history_path, 'w') as fp:
            json.dump(history_store, fp)

        # Save the model state
        torch.save(model_instance.state_dict(), model_path)

        print(f'Saved {model_name} state to {model_path}')
        print(f'Saved {model_name} results to {history_path}')
        
    def train_model(self, model_name, model_instance):
        print(f'Starting training for {model_name}...')
        
        model_instance = model_instance.to(config.device)  ######### NEW

        optimizer = torch.optim.Adam(model_instance.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        fitter = ModelFitter(
            epochs=self.epochs,
            model=model_instance,
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=scheduler
        )
            
        history_store = fitter.fit()
        
        self.save_train_results(model_name, model_instance, history_store)

        print(f'Completed training for {model_name}')
        print('----------------------------------------')

    def run(self):
        for model_name, model_instance in self.model_selector.items():
            self.train_model(model_name, model_instance)
            
            
            

class ModelPredictionPipeline:
    def __init__(self, model_selector, pred_loader):
        self.model_selector = model_selector
        self.pred_loader = pred_loader
        self.model_dir = config.model_dir
        self.device = config.device

    def save_predictions(self, model_name, all_preds, all_labels):
        pred_path = os.path.join(self.model_dir, f'preds/{model_name}_Intel_predictions.json')
        with open(pred_path, 'w') as fp:
            json.dump({
                "predictions": all_preds.tolist(),
                "true_labels": all_labels.tolist()
            }, fp)
        print(f'Saved predictions for {model_name} at {pred_path}.')

    def predict(self, model_name, model_instance):
        print(f'Getting predictions for {model_name}...')
        
        model_path = os.path.join(self.model_dir, f'params/{model_name}_Intel_params_n20.pt') 
        # model_instance.load_state_dict(torch.load(model_path))
        model_instance.load_state_dict(torch.load(model_path, map_location=self.device))   # If CPU

        model_instance.to(self.device).eval()
        
        all_preds, all_labels = get_all_predictions(
            model=model_instance, 
            loader=self.pred_loader
        )
        
        # Call the save_predictions method
        self.save_predictions(model_name, all_preds, all_labels)
        
        print(f'Completed predictions for {model_name}.')
        print('----------------------------------------')

    def run(self):
        for model_name, model_instance in self.model_selector.items():
            self.predict(model_name, model_instance)




class ModelEvaluationPipeline:
    def __init__(self, model_selector, test_loader, model_files):
        self.model_selector = model_selector
        self.test_loader = test_loader
        self.model_files = model_files
        self.model_dir = config.model_dir
        self.device = config.device

    def save_evaluation_metrics(self, model_name, evaluation_metrics):
        evaluation_metrics_path = os.path.join(self.model_dir, f'evals/{model_name}_Intel_evaluation.json')
        with open(evaluation_metrics_path, 'w') as fp:
            json.dump(evaluation_metrics, fp)
        print(f'Saved evaluation metrics for {model_name} at {evaluation_metrics_path}.')

    def evaluate(self, model_name, model_arch):
        evalutation_metrics = {}
        
        # Load model state from file with map_location argument
        model_arch.load_state_dict(torch.load(self.model_files[model_name], map_location=self.device))
        model_arch.to(self.device).eval()

        # Call Evaluator Class
        evaluator = ModelEvaluator(
            model=model_arch,
            loader=self.test_loader
            )

        # performance metrics
        performance_metrics = evaluator.compute_performance_metrics()
        class_accuracies = evaluator.class_specific_accuracy()
        computational_metrics = evaluator.computational_and_efficiency_metrics()
        complexity_metrics = evaluator.performance_vs_complexity()

        # Organize all the metrics
        evalutation_metrics = {
            'Performance Metrics': performance_metrics,
            'Class Accuracy Metrics': class_accuracies,
            'Computational Metrics': computational_metrics,
            'Complexity Metrics': complexity_metrics
        }

        # Call the save_evaluation_metrics method
        self.save_evaluation_metrics(model_name, evalutation_metrics)

    def run(self):
        for model_name, model_arch in self.model_selector.items():
            self.evaluate(model_name, model_arch)

