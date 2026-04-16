import torch
from torch import nn

cross_entropy = nn.CrossEntropyLoss()


def test_loss(model, test_loader):
    model.eval()
    cml_loss = 0
    cml_items = 0
    for x, y in test_loader:
        n_items = x.shape[0]
        with torch.no_grad():
            y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        cml_loss += loss.item() * n_items
        cml_items += n_items
    model.train()
    print(f"Computed test loss over {cml_items} items")
    return cml_loss / cml_items


def test_accuracy(model, test_loader):
    model.eval()
    cml_correct = 0
    cml_items = 0
    for x, y in test_loader:
        n_items = x.shape[0]
        with torch.no_grad():
            y_pred = model(x)
        predictions = torch.argmax(y_pred, dim=1)
        cml_correct += (y == predictions).sum().item()
        cml_items += n_items
    model.train()
    print(f"Computed test accuracy over {cml_items} items")
    return cml_correct / cml_items
