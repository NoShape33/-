"""
辅助函数：计时、平均指标计算、格式化输出等
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class AverageMeter:
    """跟踪并平均指标值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """计算批次准确率"""
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct, target.size(0)


def format_time(seconds):
    """将秒数格式化"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f'{h}h{m:02d}m{s:02d}s'
    elif m > 0:
        return f'{m}m{s:02d}s'
    else:
        return f'{s}s'


def plot_training_curves(logs, save_path=None):
    """绘制训练/验证损失和准确率曲线"""
    epochs = range(1, len(logs['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, logs['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, logs['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, logs['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, logs['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def get_classification_report(y_true, y_pred, class_names):
    """生成分类报告字典"""
    return classification_report(y_true, y_pred, target_names=class_names,
                                  output_dict=True, zero_division=0)


def plot_predictions(images, true_labels, pred_labels, save_path=None,
                     n_cols=8, title='Predictions'):
    """可视化预测结果（正确/错误分类样本）"""
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'T:{true_labels[i]} P:{pred_labels[i]}', color=color,
                          fontsize=8)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
