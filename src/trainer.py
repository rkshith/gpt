import torch
import time
from src.model import GPT

class Trainer:
    def __init__(self, model, dataloader, config, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.optimizer = optimizer
        self.device = config.device

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(iter(self.dataloader))
            X, Y = X.to(self.device), Y.to(self.device)
            _, loss = self.model(X, Y)
            losses[k] = loss.item()
        self.model.train()
        return losses.mean()

    def train(self):
        self.model.to(self.device)
        iter_num = 0
        best_val_loss = 1e9
        
        print(f"Starting training on {self.device}...")
        
        while iter_num < self.config.max_iters:
            for X, Y in self.dataloader:
                X, Y = X.to(self.device), Y.to(self.device)
                
                # Forward pass
                logits, loss = self.model(X, Y)
                
                # Backward pass
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                
                if iter_num % self.config.eval_interval == 0:
                    val_loss = self.estimate_loss(self.config.eval_iters)
                    print(f"Step {iter_num}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # In production, save checkpoint here
                
                iter_num += 1
                if iter_num >= self.config.max_iters:
                    break