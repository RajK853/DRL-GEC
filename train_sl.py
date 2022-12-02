import os
import sys
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.sl.dataset import GECDataset
from src.sl.utils import process_data, collate_func
from src.utils import load_yaml, load_text, write_json, freeze_params, load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.cuda.amp.autocast()
def evaluate(model, batch, criterion):
    labels = torch.from_numpy(batch['labels']).to(device)
    logits = model(tokens=batch['tokens'])
    batch_size, seq_size, labels_size = logits.shape
    loss = criterion(logits.view(-1, labels_size), labels.view(-1))
    loss = loss.view(batch_size, seq_size).sum(dim=-1)
    return loss


def load_data(datasets, data_type, data_limit=0):
    data = []
    filename = "data.gector" if data_type == "train" else "dev.gector"
    for dataset in tqdm(datasets, desc=f"Loading {data_type} datasets", total=len(datasets)):
        data_path = f"../data/processed/{dataset}/{filename}"
        data.extend(load_text(data_path))
    if (data_limit > 0) and (len(data) > data_limit):
        print(f"Truncating amount of data from {len(data)} to {data_limit}")
        data = data[:data_limit]
    print(f"Total number of sentences: {len(data)}")
    return data


def load_and_process_data(
        datasets,
        label_vocab,
        batch_size,
        accumulation_size,
        keep_corrects=True,
        num_workers=None,
        data_limit=0,
):
    train_data = load_data(datasets, "train", data_limit=data_limit)
    val_data = load_data(datasets, "validation")
    train_tokens, train_labels = process_data(train_data, label_vocab, keep_corrects=keep_corrects)
    val_tokens, val_labels = process_data(val_data, label_vocab, keep_corrects=True)
    label2index = {label: i for i, label in enumerate(label_vocab)}
    train_dataset = GECDataset(train_tokens, train_labels, label2index)
    val_dataset = GECDataset(val_tokens, val_labels, label2index)
    train_batch_size = int(batch_size / accumulation_size)
    train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_func,
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_func,
    )
    return train_loader, val_loader


def main(
        cold_lr,
        warm_lr,
        dropout,
        num_epochs,
        cold_epochs,
        patience,
        batch_size,
        accumulation_size,
        num_workers,
        data_limit,
        keep_corrects,
        datasets,
        label_path,
        model_path,
        log_dir,
        meta_data,
):
    train_type = "pretrain" if model_path is None else "finetune"
    current_datetime = datetime.now().strftime("%d_%m_%Y_%H:%M")
    exp_log_dir = os.path.join(log_dir, f"{train_type}_rl_{current_datetime}")
    # Load labels
    label_vocab = load_text(label_path)
    # Load and process datasets
    train_loader, val_loader = load_and_process_data(
        datasets,
        label_vocab,
        batch_size,
        accumulation_size,
        keep_corrects=keep_corrects,
        num_workers=num_workers,
        data_limit=data_limit,
    )
    # Load model
    model_name = "roberta-base"
    tokenizer_config = {"use_fast": True}
    transformer_config = {"output_attentions": False}
    model = load_model(
            model_name=model_name,
            model_path=model_path,
            num_labels=len(label_vocab),
            tokenizer_config=tokenizer_config,
            transformer_config=transformer_config,
            local_files_only=True,
    ).to(device)
    freeze_params(model.encoder, requires_grad=False)  # Freeze encoder model
    model.train()
    # Load loss function, optimizer and gradient scaler
    lr = cold_lr
    criterion = nn.CrossEntropyLoss(reduction="none")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    grad_scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=exp_log_dir)
    write_json(os.path.join(exp_log_dir, "meta.json"), meta_data)
    # Log hyperparameters
    writer.add_scalar("hyperparameters/dropout", dropout, 0)
    writer.add_scalar("hyperparameters/patience", patience, 0)
    writer.add_scalar("hyperparameters/batch_size", batch_size, 0)
    writer.add_scalar("hyperparameters/cold_epochs", cold_epochs, 0)
    writer.add_scalar("hyperparameters/keep_corrects", int(keep_corrects), 0)
    writer.add_scalar("hyperparameters/accumulation_size", accumulation_size, 0)
    writer.add_scalar("hyperparameters/uses_CE_loss", int(isinstance(criterion, nn.CrossEntropyLoss)), 0)
    # Evaluate the model
    with torch.no_grad():
        val_losses = [evaluate(model, batch, criterion) for batch in val_loader]
        val_loss = torch.cat(val_losses).mean()
        writer.add_scalar("sl/validation_loss", val_loss, 0)
    epochs_since_improvement = 0
    best_val_score = val_loss
    train_size = len(train_loader)
    for epoch in tqdm(range(num_epochs), desc="Training", total=num_epochs):
        if epoch == cold_epochs:  # Unfreeze encoder model at the end of the cold epoch
            lr = warm_lr
            freeze_params(model.encoder, requires_grad=True, optim=optim, lr=lr)
        step_offset = epoch * train_size
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}", total=len(train_loader)):
            loss = evaluate(model, batch, criterion)
            loss = loss.mean()
            grad_scaler.scale(loss / accumulation_size).backward()
            if ((i + 1) % accumulation_size) == 0:
                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad()
                torch.cuda.empty_cache()
            current_step = step_offset + i
            writer.add_scalar("sl/lr", lr, current_step)
            writer.add_scalar("sl/train_loss", loss, current_step)
        # Evaluate model
        with torch.no_grad():
            val_losses = [evaluate(model, batch, criterion) for batch in val_loader]
            val_loss = torch.cat(val_losses).mean()
        writer.add_scalar("sl/validation_loss", val_loss, epoch+1)
        if val_loss <= best_val_score:
            best_val_score = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(exp_log_dir, "model-best.pt"))  # Save best model
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print("Early stopping!")
                break
    torch.save(model.state_dict(), os.path.join(exp_log_dir, "model-last.pt"))


if __name__ == "__main__":
    config_path = sys.argv[1]
    params = load_yaml(config_path)
    main(**params)
