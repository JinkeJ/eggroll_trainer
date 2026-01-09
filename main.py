import torch
import torch.nn as nn

from torchvision import datasets, transforms

import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from eggroll import *
from model import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    return train_loader, test_loader

def fitness(model, data, target):
    output = model(data)
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    total = target.size(0)
    return correct / total

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total

def train_eggroll(params, verbose=False):
    train_loader, test_loader = get_dataloaders(params['batch_size'])
    with torch.no_grad():
        model = SimpleMLP().to(DEVICE)
    if verbose:
        acc = evaluate(model, test_loader, DEVICE)
        print(f" Epoch 0/{params['epochs']}: {acc:.4f}")

    optimizer = EggrollOptimizer(
        model=model,
        fitness_fn=fitness,
        population=params['population'],
        sigma=params['sigma'],
        lr=params['lr'],
        rank=params['rank'],
        device=DEVICE,
    )

    best_acc = float('-inf')
    for epoch in range(params['epochs']):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.step(data, target)

        if verbose:
            acc = evaluate(model, test_loader, DEVICE)
            print(f" Epoch {epoch+1}/{params['epochs']}: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'model.pth')

    return evaluate(model, test_loader, DEVICE)

def train_sgd(params, verbose=False):
    train_loader, test_loader = get_dataloaders(params['batch_size'])
    model = SimpleMLP().to(DEVICE)
    if verbose:
        acc = evaluate(model, test_loader, DEVICE)
        print(f" Epoch 0/{params['epochs']}: {acc:.4f}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(params['epochs']):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if verbose:
            acc = evaluate(model, test_loader, DEVICE)
            print(f" Epoch {epoch+1}/{params['epochs']}: {acc:.4f}")

    return evaluate(model, test_loader, DEVICE)

def main():
    EGGROLL_PARAMS = {
        'epochs': 100,          # Generations
        'batch_size': 32,
        'rank': 1,             # Low-rank parameter r
        'population': 256,    # Moderate N (paper recommends ≲10³ but larger helps)
        'sigma': 0.1,         # ES perturbation scale σ
        'lr': 0.01,            # Learning rate α (absorbs 1/σ scaling)
    }
    SGD_PARAMS = {
        'epochs': 10,          # Generations
        'batch_size': 256,
    }
    print(f"Device: {DEVICE}")
    print(f"Hyperparameters: {EGGROLL_PARAMS}")

    set_seed(42)

    acc = train_sgd(SGD_PARAMS, True)
    print(f"SGD Acc: {acc:.4f}")
	
    acc = train_eggroll(EGGROLL_PARAMS, True)
    print(f"EGGROLL Acc: {acc:.4f}")

if __name__ == "__main__":
    main()