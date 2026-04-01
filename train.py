import torch
from configs.config import GPTConfig
from src.model import GPT
from src.data_loader import get_dataloader
from src.trainer import Trainer

def main():
    # Load Data (Example: Shakespeare)
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    config = GPTConfig()
    # Update vocab size based on data
    unique_chars = set(text)
    config.vocab_size = len(unique_chars)
    
    train_loader, dataset = get_dataloader(text, config.block_size, config.batch_size, config.device)
    
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    trainer = Trainer(model, train_loader, config, optimizer)
    trainer.train()
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()