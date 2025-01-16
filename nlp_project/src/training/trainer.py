import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, train_dataloader, valid_dataloader):
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        total_steps = len(train_dataloader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            self._train_epoch(train_dataloader, optimizer, scheduler)
            valid_loss, valid_acc = self._evaluate(valid_dataloader)
            print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.4f}")

    def _train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Average training loss: {avg_loss:.4f}")

    def _evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(predictions) == np.array(actual_labels))
        
        return avg_loss, accuracy 