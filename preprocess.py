import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from shared.dataset import ShakeSpeare


def use_device(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print("------ Using cuda ------")
    elif use_mps:
        device = torch.device("mps")
        print("------ Using mps ------")
    else:
        device = torch.device("cpu")
        print("------ Using cpu ------")
    return device


def preprocess(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    current_loc = os.path.dirname(os.path.abspath(__file__))
    if args.dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
        )
    elif args.dataset == "fashion":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = datasets.FashionMNIST(
            os.path.join(current_loc, "data", "fashion-mnist"),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.FashionMNIST(
            os.path.join(current_loc, "data", "fashion-mnist"),
            train=False,
            download=True,
            transform=transform,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
        )
    elif args.dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            os.path.join(current_loc, "data", "cifar10"),
            train=True,
            transform=transform,
            download=True,
        )
        test_dataset = datasets.CIFAR10(
            os.path.join(current_loc, "data", "cifar10"),
            train=False,
            transform=transform,
            download=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
        )
    elif args.dataset == "shakespeare":
        train_dataset = ShakeSpeare(train=True)
        test_dataset = ShakeSpeare(train=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")
    return train_dataset, test_loader, use_device(args)
