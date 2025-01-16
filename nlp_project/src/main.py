import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config.config import ModelConfig, DataConfig
from data.dataset import MultilingualTextDataset
from models.classifier import MultilingualTextClassifier
from training.trainer import ModelTrainer

def main():
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Veri yükleme
    train_df = pd.read_csv(data_config.train_path)
    valid_df = pd.read_csv(data_config.valid_path)
    
    # Tokenizer yükleme
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    # Dataset oluşturma
    train_dataset = MultilingualTextDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=model_config.max_length
    )
    
    valid_dataset = MultilingualTextDataset(
        texts=valid_df['text'].values,
        labels=valid_df['label'].values,
        tokenizer=tokenizer,
        max_length=model_config.max_length
    )
    
    # Dataloader oluşturma
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=model_config.batch_size
    )
    
    # Model oluşturma
    model = MultilingualTextClassifier(
        model_name=model_config.model_name,
        num_labels=model_config.num_labels
    )
    
    # Eğitim
    trainer = ModelTrainer(model, model_config)
    trainer.train(train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()