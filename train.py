import argparse
import pandas as pd
import tiktoken
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from LanguageModel_OnlineMovieComments.download_gpt_weights import download_model_weights
from LanguageModel_OnlineMovieComments.models import GPTModel, load_weights_into_gpt

from train_utils import calc_loss_batch, calc_loss_loader, calc_accuracy_loader, evaluate_model

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
    
def instantiate_model(choose_model, load_weights):

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # 上下文长短
        "drop_rate": 0.0,       
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    }

    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG)

    if load_weights:
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_model_weights(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    model.eval()
    return model

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  
            loss = calc_loss_batch(input_batch, target_batch, model, device,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  
            optimizer.step()  
            examples_seen += input_batch.shape[0]  
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "Choose a GPT model to use: 'gpt2-small (124M)', 'gpt2-medium (355M)'"
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "Which token to train. Options: 'first', 'last'."
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action='store_true',
        default=False,
        help=(
            "Average the output embeddings from all tokens instead of using"
            " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="256",
        help=(
            ""
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            ""
        )
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=(
            ""
        )
    )
    args = parser.parse_args()

    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    else:
        raise ValueError("Invalid --trainable_token_pos argument")

    ###############################   load model   ###############################

    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("Invalid --weights argument.")

    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        param.requires_grad = False

    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")

    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    model.to(device)

    ############################### dataloaders ###############################

    base_path = Path(".")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None
    if args.context_length == "model_context_length":
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            max_length = int(args.context_length)
        except ValueError:
            raise ValueError("Invalid --context_length argument")

    if train_dataset is None:
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

    ############################### Train 

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        average_embeddings=args.average_embeddings
    )

    ############################### Evaluate model

    print("\nEvaluating on the full datasets ...\n")

    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
