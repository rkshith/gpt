import torch
from configs.config import GPTConfig
from src.model import GPT
from src.data_loader import TextDataset

def main():
    # Load context
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize dataset to get tokenizer
    dummy_dataset = TextDataset(text, 256)
    
    config = GPTConfig()
    config.vocab_size = dummy_dataset.vocab_size
    
    model = GPT(config)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    
    start = "Hello, "
    context = torch.tensor(dummy_dataset.encode(start), dtype=torch.long).unsqueeze(0)
    
    generated = model.generate(context, max_new_tokens=500)
    print(dummy_dataset.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()