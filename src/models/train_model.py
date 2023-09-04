from config import Config

import os
import torch
import torch.nn as nn
import torch.nn.functional as F 

import timm
from transformers import AutoImageProcessor, ResNetForImageClassification

config = Config()


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch[0].to(config.device), batch[1].to(config.device)  # Move data to config.device
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch[0].to(config.device), batch[1].to(config.device)  # Move data to config.device
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        _, preds = torch.max(out, dim=1)      # Get the predicted labels
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class SimpleCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(82944, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6)
        self.dropout4 = nn.Dropout(0.5)

    def conv_block(self, x, conv1, conv2, pool, dropout):
        x = F.relu(conv1(x))
        x = dropout(pool(F.relu(conv2(x))))
        return x

    def forward(self, x):
        x = self.conv_block(x, self.conv1_1, self.conv1_2, self.pool1, self.dropout1)
        x = self.conv_block(x, self.conv2_1, self.conv2_2, self.pool2, self.dropout2)
        x = self.conv_block(x, self.conv3_1, self.conv3_2, self.pool3, self.dropout3)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout4(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class MyEfficientNet(ImageClassificationBase):
    def __init__(self, model_name='efficientnet_b0', num_classes=6):
        super(MyEfficientNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)    
    
    
class PreTrainedEfficientNet(ImageClassificationBase):
    def __init__(self, model_name='efficientnet_b0', num_classes=6):
        super(PreTrainedEfficientNet, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class HuggingfaceResNet(ImageClassificationBase):
    def __init__(self, model_name='microsoft/resnet-50', num_classes=6):
        super(HuggingfaceResNet, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetForImageClassification.from_pretrained(model_name)

    def forward(self, x):
        return self.model(x).logits


class ModelFitter():
    def __init__(self, epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
        self.epochs = epochs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def fit(self):
        history = []  # for recording epoch-wise results
        
        for epoch in range(self.epochs):
            
            # Training phase
            self.model.train()
            train_losses = []
            for batch in self.train_loader:
                loss = self.model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # Validation phase
            self.model.eval()
            outputs = [self.model.validation_step(batch) for batch in self.val_loader]
            validation_result = self.model.validation_epoch_end(outputs)
            result = {**validation_result, 'train_loss': torch.stack(train_losses).mean().item()}
            self.model.epoch_end(epoch, result)
            history.append(result)
            self.scheduler.step()

        return history
    

        