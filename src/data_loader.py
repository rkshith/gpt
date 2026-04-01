import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = text
        self.block_size = block_size
        # Simple Char-Level Tokenizer
        self.stoi = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
        self.itos = {i: ch for i, ch in enumerate(sorted(list(set(text))))}
        self.vocab_size = len(self.stoi)
        self.encoded_data = self.encode(self.data)

    def __len__(self):
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        x = self.encoded_data[idx:idx + self.block_size]
        y = self.encoded_data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

def get_dataloader(text, block_size, batch_size, device):
    dataset = TextDataset(text, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset