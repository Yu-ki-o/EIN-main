"""Small metric helpers for the TCSR prototype."""

import numpy as np
import torch


def accuracy_score(prediction, target):
    if torch.is_tensor(prediction) or torch.is_tensor(target):
        prediction = _to_tensor_prediction(prediction, target)
        target = _to_tensor(target, prediction.device).view(-1).long()
        if target.numel() == 0:
            return 0.0
        return float((prediction.view(-1) == target).float().mean().item())

    prediction = _to_numpy(prediction).reshape(-1)
    target = _to_numpy(target).reshape(-1)
    if target.size == 0:
        return 0.0
    return float((prediction == target).mean())


def macro_f1_score(prediction, target, num_classes=None):
    if torch.is_tensor(prediction) or torch.is_tensor(target):
        prediction = _to_tensor_prediction(prediction, target).view(-1)
        target = _to_tensor(target, prediction.device).view(-1).long()
        if target.numel() == 0:
            return 0.0
        if num_classes is None:
            max_label = torch.cat((prediction, target)).max()
            num_classes = int(max_label.item()) + 1
        classes = torch.arange(int(num_classes), device=prediction.device)
        pred_pos = prediction.unsqueeze(-1) == classes
        true_pos = target.unsqueeze(-1) == classes
        tp = (pred_pos & true_pos).sum(dim=0).float()
        fp = (pred_pos & ~true_pos).sum(dim=0).float()
        fn = (~pred_pos & true_pos).sum(dim=0).float()
        precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
        recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
        f1 = torch.where(
            precision + recall > 0,
            2.0 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )
        return float(f1.mean().item()) if f1.numel() else 0.0

    prediction = _to_numpy(prediction).reshape(-1)
    target = _to_numpy(target).reshape(-1)
    if target.size == 0:
        return 0.0
    if num_classes is None:
        max_label = 0
        if prediction.size:
            max_label = max(max_label, int(prediction.max()))
        if target.size:
            max_label = max(max_label, int(target.max()))
        num_classes = max_label + 1

    f1_values = []
    for class_id in range(int(num_classes)):
        pred_pos = prediction == class_id
        true_pos = target == class_id
        tp = float(np.logical_and(pred_pos, true_pos).sum())
        fp = float(np.logical_and(pred_pos, ~true_pos).sum())
        fn = float(np.logical_and(~pred_pos, true_pos).sum())
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_values)) if f1_values else 0.0


def classification_metrics(logits_or_prediction, target, num_classes=None):
    if torch.is_tensor(logits_or_prediction):
        if logits_or_prediction.dim() > 1:
            prediction = logits_or_prediction.argmax(dim=-1)
        else:
            prediction = logits_or_prediction
    else:
        prediction = logits_or_prediction
    return {
        "accuracy": accuracy_score(prediction, target),
        "macro_f1": macro_f1_score(
            prediction,
            target,
            num_classes=num_classes,
        ),
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    return torch.as_tensor(value, device=device)


def _to_tensor_prediction(prediction, target):
    if torch.is_tensor(prediction):
        prediction = prediction.detach()
        if prediction.dim() > 1:
            prediction = prediction.argmax(dim=-1)
        return prediction.long()
    device = target.device if torch.is_tensor(target) else torch.device("cpu")
    return torch.as_tensor(prediction, device=device).view(-1).long()
