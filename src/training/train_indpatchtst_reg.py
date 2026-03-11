import torch


def train_epoch(model, dataloader, optimizer, criterion, device="cuda"):
    model.to(device)
    model.train()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        pred = model(past)
        loss = criterion(pred, future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device="cuda"):
    model.to(device)
    model.eval()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        loss = criterion(model(past), future)
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


def train_and_valid_loop(
    model,
    train_dl,
    valid_dl,
    optimizer,
    criterion,
    n_epochs,
    device="cuda",
    scheduler=None,
):
    logs = {"train_loss": [], "valid_loss": []}
    best_loss, best_state = float("inf"), None
    for epoch in range(n_epochs):
        tr = train_epoch(model, train_dl, optimizer, criterion, device)
        vl = eval_epoch(model, valid_dl, criterion, device)
        if scheduler:
            scheduler.step()
        logs["train_loss"].append(tr)
        logs["valid_loss"].append(vl)
        print(f"Epoch {epoch + 1:02d} | train={tr:.4f} | valid={vl:.4f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return logs
