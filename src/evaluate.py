"""
评估与可视化：定量评估、定性分析、错误案例分析、特征可视化
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json

from data_loader import get_test_loader_for_eval, plot_dataset_samples
from model import BasicCNN, MiniResNet, count_parameters
from utils import (AverageMeter, accuracy, plot_training_curves,
                    plot_confusion_matrix, get_classification_report,
                    plot_predictions)


def load_model(model_path, model_type='BasicCNN'):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_map = {'BasicCNN': BasicCNN, 'MiniResNet': MiniResNet}
    model = model_map[model_type]().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device


def quantitative_eval(model, device, save_dir='figures'):
    """定量评估：混淆矩阵、分类报告、Loss/Acc曲线"""
    test_loader = get_test_loader_for_eval(batch_size=1000)

    all_preds = []
    all_labels = []
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        preds = output.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = [str(i) for i in range(10)]
    cm = plot_confusion_matrix(all_labels, all_preds, class_names,
                                save_path=f'{save_dir}/confusion_matrix.png')

    report = get_classification_report(all_labels, all_preds, class_names)
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")
    print(f"\nClassification Report:")
    for class_name in class_names:
        metrics = report[class_name]
        print(f"  Digit {class_name}: "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"F1={metrics['f1-score']:.4f}")

    print(f"\n  Macro Avg: P={report['macro avg']['precision']:.4f}, "
          f"R={report['macro avg']['recall']:.4f}, "
          f"F1={report['macro avg']['f1-score']:.4f}")

    with open(f'{save_dir}/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return all_preds, all_labels, overall_acc


def visualize_error_cases(model, device, save_dir='figures', n_errors=16):
    """可视化错误分类样本"""
    test_loader = get_test_loader_for_eval(batch_size=1)
    errors = []

    for data, target in test_loader:
        data_dev = data.to(device)
        output = model(data_dev)
        pred = output.argmax(dim=1).item()
        if pred != target.item():
            errors.append((data.squeeze().numpy(), target.item(), pred))
        if len(errors) >= n_errors:
            break

    if errors:
        images = [e[0] for e in errors]
        true_labels = [e[1] for e in errors]
        pred_labels = [e[2] for e in errors]
        plot_predictions(images, true_labels, pred_labels,
                         save_path=f'{save_dir}/error_cases.png',
                         title='Error Case Analysis (True vs Predicted)')


def visualize_correct_cases(model, device, save_dir='figures', n_samples=16):
    """可视化正确分类样本"""
    test_loader = get_test_loader_for_eval(batch_size=1)
    correct = []

    for data, target in test_loader:
        data_dev = data.to(device)
        output = model(data_dev)
        pred = output.argmax(dim=1).item()
        if pred == target.item():
            correct.append((data.squeeze().numpy(), target.item(), pred))
        if len(correct) >= n_samples:
            break

    if correct:
        images = [c[0] for c in correct]
        true_labels = [c[1] for c in correct]
        pred_labels = [c[2] for c in correct]
        plot_predictions(images, true_labels, pred_labels,
                         save_path=f'{save_dir}/correct_predictions.png',
                         title='Correct Predictions')


def visualize_conv_filters(model, device, save_dir='figures', layer_name='conv1'):
    """可视化第一层卷积核"""
    conv_layer = getattr(model, layer_name, None)
    if conv_layer is None:
        print(f"Layer {layer_name} not found")
        return

    weights = conv_layer.weight.data.cpu().numpy()
    n_filters = weights.shape[0]
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2 * n_rows))
    axes = axes.flatten()

    vmin, vmax = weights.min(), weights.max()
    for i in range(n_filters):
        axes[i].imshow(weights[i, 0], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{layer_name} Filters')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conv1_filters.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_feature_maps(model, device, save_dir='figures'):
    """可视化中间层特征图（复现真实前向传播路径）"""
    test_loader = get_test_loader_for_eval(batch_size=1)
    data, _ = next(iter(test_loader))
    data = data.to(device)

    layer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach().cpu()
        return hook

    hooks = []
    for name, layer in model.named_children():
        if name in ['conv1', 'conv2', 'conv3']:
            hooks.append(layer.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(data)

    for hook in hooks:
        hook.remove()

    for layer_name, feat_map in layer_outputs.items():
        n_channels = min(feat_map.shape[1], 8)
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i in range(n_channels):
            axes[i].imshow(feat_map[0, i], cmap='viridis')
            axes[i].axis('off')
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Feature Maps - {layer_name}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_map_{layer_name}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_comparison_bar(all_results_path='outputs/all_results.json',
                         save_dir='figures'):
    """绘制对照实验对比柱状图"""
    with open(all_results_path, 'r') as f:
        all_results = json.load(f)

    exp_names = list(all_results.keys())
    best_val_accs = [max(logs['val_acc']) for logs in all_results.values()]
    best_test_accs = []

    for i, key in enumerate(exp_names):
        model_type = key.split('_')[0]
        model_path = f"outputs/best_model_{key}.pth"
        try:
            model, device = load_model(model_path, model_type)
            _, _, test_acc = quantitative_eval(
                model, device, save_dir='figures')
            best_test_accs.append(test_acc)
        except FileNotFoundError:
            best_test_accs.append(0)

    x = np.arange(len(exp_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, best_val_accs, width, label='Best Val Acc',
                    color='steelblue')
    bars2 = ax.bar(x + width/2, best_test_accs, width, label='Test Acc',
                    color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison: Validation vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def main():
    print("="*60)
    print("Model Evaluation and Visualization")
    print("="*60)

    model_path = "outputs/best_model_BasicCNN_adam.pth"
    model, device = load_model(model_path, 'BasicCNN')
    print(f"Loaded model on {device}")
    print(f"Parameters: {count_parameters(model):,}")

    print("\n[1/6] Plotting dataset samples...")
    plot_dataset_samples(save_path='figures/dataset_samples.png')

    print("\n[2/6] Quantitative evaluation...")
    all_preds, all_labels, overall_acc = quantitative_eval(
        model, device, save_dir='figures')

    print("\n[3/6] Visualizing correct predictions...")
    visualize_correct_cases(model, device, save_dir='figures')

    print("\n[4/6] Visualizing error cases...")
    visualize_error_cases(model, device, save_dir='figures')

    print("\n[5/6] Visualizing conv filters...")
    visualize_conv_filters(model, device, save_dir='figures')

    print("\n[6/6] Visualizing feature maps...")
    visualize_feature_maps(model, device, save_dir='figures')

    if os.path.exists('outputs/all_results.json'):
        print("\n[Bonus] Plotting model comparison...")
        plot_comparison_bar(save_dir='figures')

    print(f"\n{'='*60}")
    print("Evaluation complete. All figures saved to figures/")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    import os
    main()
