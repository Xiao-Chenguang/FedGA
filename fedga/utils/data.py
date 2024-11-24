import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from datasets import load_dataset


def fed_dataloader(args):
    ds, index_client, test_ds, classes = ib_fed_ds(
        args.DATA.NAME,
        args.FL.CLIENT.TOTAL,
        args.DATA.IB.ALPHA,
        args.DATA.PATH,
    )
    shape = (
        list(ds[0]["image"].shape) if isinstance(ds[0], dict) else list(ds[0][0].shape)
    )

    fed_ds = [ClientDataset(ds, idx, args) for idx in index_client]

    fed_dl = [
        DataLoader(
            dataset,
            batch_size=args.DATA.TRAIN.BATCH,
            shuffle=True,
            num_workers=args.DATA.TRAIN.WORKERS,
        )
        for dataset in fed_ds
    ]
    test_dl = DataLoader(
        test_ds,
        batch_size=args.DATA.TEST.BATCH,
        num_workers=args.DATA.TEST.WORKERS,
        shuffle=False,
    )
    # log data info into args
    args.DATA.SHAPE = shape
    args.DATA.CLASSES = classes
    args._box_config["frozen_box"] = True
    return fed_dl, test_dl, shape, classes


def ib_fed_ds(dataSet, n_client=100, alpha=1, ds_path="./datasets"):
    dss = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
        "SVHN": datasets.SVHN,
    }

    args = {"download": True, "transform": ToTensor()}
    if dataSet == "SVHN":
        ds = dss[dataSet](ds_path, split="train", **args)
        targets = ds.labels.tolist()
        test_ds = dss[dataSet](ds_path, split="test", **args)
    elif dataSet in dss:
        ds = dss[dataSet](root=ds_path, train=True, **args)
        test_ds = dss[dataSet](root=ds_path, train=False, **args)
        targets = ds.targets if isinstance(ds.targets, list) else ds.targets.tolist()
    elif dataSet == "Tiny-ImageNet":

        def process(examples):
            ts = ToTensor()
            examples["image"] = [ts(img).expand(3, -1, -1) for img in examples["image"]]
            return examples

        name = "zh-plus/tiny-imagenet"
        cache_dir = ds_path + "/tiny-imagenet"
        ds = load_dataset(name, cache_dir=cache_dir, split="train")
        targets = ds["label"]
        ds.set_transform(process)
        test_ds = load_dataset(name, cache_dir=cache_dir, split="valid")
        test_ds.set_transform(process)
    else:
        raise NotImplementedError(f"Dataset {dataSet} not included!")

    n_class = len(set(targets))
    index_client = naive_dir_ib_index(targets, n_client, n_class, alpha)

    return ds, index_client, test_ds, n_class


def naive_dir_ib_index(targets, n_client, n_class, alpha=1):
    targets = np.array(targets)
    samples = [[] for _ in range(n_client)]
    classes_index = [
        np.squeeze(np.argwhere(targets == i), axis=-1).tolist() for i in range(n_class)
    ]
    for i in range(n_class):
        np.random.shuffle(classes_index[i])
    sizes = np.array([len(row) for row in classes_index])

    ratios = np.random.dirichlet(np.ones(n_class) * alpha, size=n_client)
    class_ratios = ratios.sum(axis=0)
    nums = (sizes / class_ratios).min() * ratios
    nums = nums.round().astype(int)
    print(nums)

    for i in range(n_class):
        start = 0
        for j in range(n_client):
            samples[j].append(classes_index[i][start : start + nums[j, i]])
            start += nums[j, i]
    return samples


class ClientDataset(Dataset):
    def __init__(self, dataset, index, args=None) -> None:
        """
        dataset: dataset
        index: list of list of class index
        """
        super().__init__()
        self.dataset = dataset
        self.class_idx = index
        self.index = sum(index, [])
        self.label_dist = [len(x) for x in index]

    def __getitem__(self, idx):
        return self.dataset[self.index[idx]]

    def __len__(self):
        return len(self.index)
