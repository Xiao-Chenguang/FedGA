from torchmetrics.classification import MulticlassF1Score


def ib_metrics(pred, gtrue, device, classes=10):
    """
    pred: tensor of logits(sum to 1) of shpape [batch_size, num_class]
    gtrue: tensor of ground truth, shape [batch_size,]
    classes: number of classes
    """
    acc_mic = (pred.argmax(axis=-1) == gtrue).sum().item() / gtrue.shape[0]
    f1_mac = (
        MulticlassF1Score(num_classes=classes, average="macro")
        .to(device)(pred, gtrue)
        .item()
    )

    return {
        "V-Acc": acc_mic,
        "V-F1": f1_mac,
    }
