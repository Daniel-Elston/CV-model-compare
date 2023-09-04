import os
import dotenv
import yaml
import json
import torch

class Config:
    def __init__(self):
        # Load .env
        project_dir = os.getenv('PROJECT_DIR')
        dotenv_path = os.path.join(project_dir, '.env')
        dotenv.load_dotenv(dotenv_path)
        
        # Set up paths
        self.data_vault_dir = os.getenv('VAULT_DIR')
        self.db_url = os.getenv('DATABASE_URL')
        self.data_dir = os.path.join(project_dir, 'data')
        self.model_dir = os.path.join(project_dir, 'models')
        self.src_dir = os.path.join(project_dir, 'src')
        
        # Load config.yaml
        config_path = os.path.join(project_dir, 'config.yaml')
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        # Other settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")