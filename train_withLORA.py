
#####  dataloader #####
if __name__ == "__main__":
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    import pandas as pd
    import tiktoken
    class IMDBDataset(Dataset):
        def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
            self.data = pd.read_csv(csv_file)
            self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

            self.encoded_texts = [
                tokenizer.encode(text)[:self.max_length]
                for text in self.data["text"]
            ]
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

        def __getitem__(self, index):
            encoded = self.encoded_texts[index]
            label = self.data.iloc[index]["label"]
            return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

        def __len__(self):
            return len(self.data)

        def _longest_encoded_length(self, tokenizer):
            max_length = 0
            for text in self.data["text"]:
                encoded_length = len(tokenizer.encode(text))
                if encoded_length > max_length:
                    max_length = encoded_length
            return max_length
    
    base_path = Path(".")

    tokenizer = tiktoken.get_encoding("gpt2")

    context_length = 256
    max_length = int(context_length)
        
    train_dataset = IMDBDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
    val_dataset = IMDBDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDBDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )


##### build model #####
if __name__ == "__main__":
    from download_gpt_weights import download_model_weights
    from models import GPTModel, load_weights_into_gpt

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_model_weights(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

##### model output set #####
if __name__ == "__main__":
    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

##### lora layer #####
if __name__ == "__main__":
    import math
    class LoRALayer(torch.nn.Module):
        def __init__(self, in_dim, out_dim, rank, alpha):
            super().__init__()
            self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
            torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
            self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
            self.alpha = alpha

        def forward(self, x):
            x = self.alpha * (x @ self.A @ self.B)
            return x
    
    class LinearWithLoRA(torch.nn.Module):
        def __init__(self, linear, rank, alpha):
            super().__init__()
            self.linear = linear
            self.lora = LoRALayer(
                linear.in_features, linear.out_features, rank, alpha
            )

        def forward(self, x):
            return self.linear(x) + self.lora(x)
    
    def replace_linear_with_lora(model, rank, alpha):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
            else:
                replace_linear_with_lora(module, rank, alpha)

##### model set and equiped with LORA #####
if __name__ == "__main__":
    for param in model.parameters():
        param.requires_grad = False

    replace_linear_with_lora(model, rank=16, alpha=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # print(model)

##### train #####
if __name__ == "__main__":
    import time
    from train_utils import train_classifier_simple, calc_accuracy_loader

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 2
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

##### eval & plot #####
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()

        ax2 = ax1.twiny()  
        ax2.plot(examples_seen, train_values, alpha=0)  
        ax2.set_xlabel("Examples seen")

        fig.tight_layout()  
        plt.savefig(f"{label}-plot.pdf")
        plt.show()
    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")