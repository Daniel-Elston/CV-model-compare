from config import Config

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision.utils import make_grid

config = Config()
    
###########################################################


class DistributionVisualizer:
    def __init__(self, df, df_train, df_test, df_pred):
        self.df = df
        self.df_train = df_train
        self.df_test = df_test
        self.df_pred = df_pred
    
    def dataset_distributions(self):
        print(self.df.Segment.value_counts())

        # Get the distribution of image categories within each segment
        train_distribution = self.df_train['Category'].value_counts(normalize=True)
        test_distribution = self.df_test['Category'].value_counts(normalize=True)
        pred_distribution = self.df_pred['Category'].value_counts(normalize=True)

        # Put into a dictionary
        distribution_dict = {'train': train_distribution,
                            'test': test_distribution,
                            'pred': pred_distribution}

        print(distribution_dict)



class ImageVisualizer:
    def __init__(self, data_loader, model=None):
        self.data_loader = data_loader
        self.model = model

    def display_single_image(self):
        # Create an iterator and fetch the first image from the first batch
        images, labels = next(iter(self.data_loader))
        image, label = images[0], labels[0]

        # Convert the image tensor to a numpy array and display it
        plt.imshow(image.permute(1, 2, 0))
        plt.title(label)
        plt.show()
        
    def display_single_image(self):
        # Create an iterator and fetch the first image from the first batch
        images, labels = next(iter(self.data_loader))
        image, label = images[0], labels[0]

        # Convert the image tensor to a numpy array and display it
        plt.imshow(image.permute(1, 2, 0))
        plt.title(label)
        plt.show()

    def display_batch_images(self):
        images, labels = next(iter(self.data_loader))

        num_images = 4

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i in range(num_images):
            ax = axes[i]
            ax.imshow(images[i].permute(1, 2, 0))
            ax.set_title(f'Label: {labels[i].item()}')
        plt.tight_layout()
        plt.show()

    def show_batch(self):
        """Plot images grid of single batch"""
        for images, labels in self.data_loader:
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break

        plt.show()


    def display_images_with_predictions(self, preds, true_labels):
        images, _ = next(iter(self.data_loader))
        
        images = images[:16]
        preds = preds[:16]
        true_labels = true_labels[:16]

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(images[i].permute(1, 2, 0))
            title = f"True: {true_labels[i]}\nPred: {preds[i]}"
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
def plot_confusion_matrices(data_dict):
    """
    Plot confusion matrices given a dictionary of predictions and true labels.
    """
    n = len(data_dict)
    fig, axes = plt.subplots(n, 1, figsize=(15, 5 * n))

    for ax, (title, data) in zip(axes, data_dict.items()):
        sns.heatmap(pd.crosstab(data[0], data[1]), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()