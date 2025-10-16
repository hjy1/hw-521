import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# This script implements a small PGD adversarial training experiment on a
# synthetic dataset. It trains a baseline (standard) model and adversarially
# trained models for several epsilons and reports standard and robust accuracy.

# reproducibility
torch.manual_seed(13)
np.random.seed(13)


def create_model():
    return nn.Sequential(
        nn.Linear(10, 10, bias=False),
        nn.ReLU(),
        nn.Linear(10, 10, bias=False),
        nn.ReLU(),
        nn.Linear(10, 3, bias=False),
    )


def pgd_attack(model, x, y, eps, alpha, iters, device='cpu', random_start=True):
    """Perform an L-inf PGD attack on the batch (x, y).

    x: tensor (B, D)
    y: tensor (B,)
    eps: scalar (max L-inf perturbation)
    alpha: step size
    iters: number of iterations
    """
    model.eval()
    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(iters):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad
            # single-step sign update
            x_adv = x_adv + alpha * grad.sign()
            # project back to L-inf ball around x
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0.0, 1.0)
        x_adv.grad = None

    return x_adv.detach()


def train_standard(model, loader, epochs=10, lr=1e-2, device='cpu'):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        # simple progress print
        if (epoch + 1) % max(1, epochs // 3) == 0:
            print(f"[standard train] epoch {epoch+1}/{epochs}, avg loss: {total_loss/len(loader.dataset):.4f}")


def train_adv(model, loader, eps, pgd_alpha, pgd_iters, epochs=10, lr=1e-2, device='cpu'):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # craft adversarial examples on-the-fly using current model
            xb_adv = pgd_attack(model, xb, yb, eps=eps, alpha=pgd_alpha, iters=pgd_iters, device=device)
            logits = model(xb_adv)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % max(1, epochs // 3) == 0:
            print(f"[adv train eps={eps:.3f}] epoch {epoch+1}/{epochs}, avg loss: {total_loss/len(loader.dataset):.4f}")


def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


def evaluate_robust(model, loader, eps, pgd_alpha, pgd_iters, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb_adv = pgd_attack(model, xb, yb, eps=eps, alpha=pgd_alpha, iters=pgd_iters, device=device)
        logits = model(xb_adv)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


def make_synthetic_data(n_train=500, n_test=200):
    # features in [0,1], 10-dim, 3 classes
    X_train = torch.rand((n_train, 10))
    y_train = torch.randint(0, 3, (n_train,))
    X_test = torch.rand((n_test, 10))
    y_test = torch.randint(0, 3, (n_test,))
    return (X_train, y_train), (X_test, y_test)


def main():
    device = 'cpu'

    # create data loaders
    (X_train, y_train), (X_test, y_test) = make_synthetic_data(n_train=500, n_test=200)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]

    # Train a standard model (baseline)
    print("Training standard baseline model...")
    std_model = create_model()
    train_standard(std_model, train_loader, epochs=12, lr=0.1, device=device)
    std_acc = evaluate(std_model, test_loader, device=device)
    print(f"Baseline standard accuracy (clean test): {std_acc*100:.2f}%")

    # Evaluate baseline robust accuracy under various epsilons
    print("Evaluating baseline under PGD attacks...")
    for eps in epsilons:
        if eps == 0.0:
            robust = std_acc
        else:
            pgd_alpha = max(1e-6, eps / 4)
            robust = evaluate_robust(std_model, test_loader, eps=eps, pgd_alpha=pgd_alpha, pgd_iters=20, device=device)
        print(f"Baseline robust acc @ eps={eps:.3f}: {robust*100:.2f}%")

    # Adversarial training for each epsilon
    results = {}
    for eps in epsilons[1:]:
        print('\n' + '='*60)
        print(f"Adversarial training with eps={eps}")
        adv_model = create_model()
        pgd_alpha = max(1e-6, eps / 4)
        # use a relatively small number of PGD iters during training for speed
        train_adv(adv_model, train_loader, eps=eps, pgd_alpha=pgd_alpha, pgd_iters=10, epochs=10, lr=0.1, device=device)
        clean_acc = evaluate(adv_model, test_loader, device=device)
        robust_acc = evaluate_robust(adv_model, test_loader, eps=eps, pgd_alpha=pgd_alpha, pgd_iters=20, device=device)
        results[eps] = (clean_acc, robust_acc)
        print(f"Adv-trained model (eps={eps:.3f}) clean acc: {clean_acc*100:.2f}%, robust acc: {robust_acc*100:.2f}%")

    print('\nSummary (eps -> clean acc, robust acc):')
    for eps, (clean_acc, robust_acc) in results.items():
        print(f"eps={eps:.3f} -> clean: {clean_acc*100:.2f}%, robust: {robust_acc*100:.2f}%")


if __name__ == '__main__':
    main()