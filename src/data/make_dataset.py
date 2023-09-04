from config import Config

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

config = Config()


class RawDataHandler:
    def __init__(self, raw_data_path):
        self.raw_data_path = os.path.join(config.data_dir, raw_data_path)
        self.df = None
        
    def fetch_raw_data(self):
        segments = ['seg_train', 'seg_test', 'seg_pred']
        image_paths = []

        for segment in segments:
            for root, dirs, files in os.walk(os.path.join(self.raw_data_path, segment, segment)):
                for file in files:
                    if file.endswith(".jpg"):
                        category = os.path.basename(root)
                        full_path = os.path.join(root, file)
                        image_paths.append((segment, category, full_path))
                        
        self.df = pd.DataFrame(image_paths, columns=["Segment", "Category", "Path"])
        return self.df

    def encode_labels(self):
        category_to_int = {
            "mountain": 0,
            "glacier": 1,
            "street": 2,
            "sea": 3,
            "forest": 4,
            "buildings": 5
        }

        # Convert string labels to integer labels
        self.df['Label'] = self.df['Category'].map(category_to_int)
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        return self.df



class DatasetHandler:
    def __init__(self, data_path):
        self.raw_data_path = os.path.join(config.data_dir, data_path)
        self.df = pd.read_parquet(self.raw_data_path, engine='pyarrow')
        self.df_train = self.df[self.df['Segment'].str.contains("seg_train")]
        self.df_test = self.df[self.df['Segment'].str.contains("seg_test")]
        self.df_pred = self.df[self.df['Segment'].str.contains("seg_pred")]

    def get_datasets(self):
        return self.df, self.df_train, self.df_test, self.df_pred
    
    def get_loaders(self, batch_size=64, shuffle=False): # Dataset already shuffled
        transform = transforms.Compose([
            transforms.Resize((150, 150)),  # Resize all images to be 150x150
            transforms.ToTensor(),
        ])
        
        train_dataset = CustomImageDataset(df=self.df_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        test_dataset = CustomImageDataset(df=self.df_test, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        
        pred_dataset = CustomImageDataset(df=self.df_pred, transform=transform)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=shuffle)
        
        return train_loader, test_loader, pred_loader


class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 2]  # Fetching the file path
        label = self.df.iloc[idx, 3].astype(int)  # Fetching the category/label
        image = Image.open(img_path)
    
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
