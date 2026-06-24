"""
训练主脚本：支持基础CNN和Mini-ResNet两种模型，
包含训练循环、验证、早停、学习率调度、日志记录
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import os
import sys
from tqdm import tqdm

from data_loader import get_dataloaders
from model import BasicCNN, MiniResNet, count_parameters
from utils import AverageMeter, accuracy


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc='Training', leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        correct, total = accuracy(output, target)
        loss_meter.update(loss.item(), total)
        acc_meter.update(100.0 * correct / total, total)
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}',
                          'acc': f'{acc_meter.avg:.2f}%'})

    return loss_meter.avg, acc_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            correct, total = accuracy(output, target)
            loss_meter.update(loss.item(), total)
            acc_meter.update(100.0 * correct / total, total)

    return loss_meter.avg, acc_meter.avg


def train_model(model, train_loader, val_loader, config, device,
                save_path='outputs/best_model.pth'):
    """
    完整训练流程，返回训练日志
    """
    criterion = nn.NLLLoss()

    optimizer_name = config.get('optimizer', 'adam')
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=config.get('weight_decay', 1e-4))
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              momentum=0.9,
                              weight_decay=config.get('weight_decay', 1e-4))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=3)

    logs = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 5)

    print(f"\n{'='*60}")
    print(f"Model: {config['model_name']} | Optimizer: {optimizer_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logs['train_loss'].append(round(train_loss, 4))
        logs['train_acc'].append(round(train_acc, 2))
        logs['val_loss'].append(round(val_loss, 4))
        logs['val_acc'].append(round(val_acc, 2))
        logs['lr'].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
            }, save_path)
            print(f"  [*] Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    return logs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=64, val_split=0.2)

    base_config = {
        'lr': 0.001,
        'epochs': 20,
        'weight_decay': 1e-4,
        'patience': 5,
    }

    experiments = [
        {**base_config, 'model_name': 'BasicCNN', 'optimizer': 'adam'},
        {**base_config, 'model_name': 'BasicCNN', 'optimizer': 'sgd',
         'lr': 0.01},
        {**base_config, 'model_name': 'MiniResNet', 'optimizer': 'adam'},
    ]

    model_map = {'BasicCNN': BasicCNN, 'MiniResNet': MiniResNet}
    all_results = {}

    for exp in experiments:
        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp['model_name']} + {exp['optimizer']}")
        print(f"{'#'*60}")

        model = model_map[exp['model_name']]().to(device)
        save_name = f"outputs/best_model_{exp['model_name']}_{exp['optimizer']}.pth"
        logs = train_model(model, train_loader, val_loader, exp,
                           device, save_path=save_name)

        exp_key = f"{exp['model_name']}_{exp['optimizer']}"
        all_results[exp_key] = logs

        with open(f"outputs/logs_{exp_key}.json", 'w') as f:
            json.dump(logs, f, indent=2)

    with open("outputs/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("All experiments complete. Results saved to outputs/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
