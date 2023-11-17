import argparse
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from agent_utils import Agent, Server
from tqdm import tqdm
from client_sampling import client_sampling
from log import log
from data_dist import DirichletSampler

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 trainning")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr", type=float, default=0.03, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument("--seed", type=int, default=420, help="random seed")
parser.add_argument("--sampling_type", type=str, default="uniform", help="")
parser.add_argument("--local_update", type=int, default=20, help="Local iterations")
parser.add_argument(
    "--num_clients", type=int, default=250, help="Total number of clients"
)
parser.add_argument("--rounds", type=int, default=5000, help="The number of rounds")
parser.add_argument("--q", type=float, default=0.5, help="Probability q")
parser.add_argument(
    "--alpha", type=float, default=0.5, help="Dirichlet Distribution parameter"
)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
current_loc = os.path.dirname(os.path.abspath(__file__))

transform_train = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = datasets.CIFAR10(
    os.path.join(current_loc, "data", "cifar10"),
    train=True,
    transform=transform_train,
    download=True,
    target_transform=None,
)

img_size = train_dataset[0][0].shape

test_dataset = datasets.CIFAR10(
    os.path.join(current_loc, "data", "cifar10"),
    train=False,
    transform=transform_test,
    download=True,
    target_transform=None,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)


class Net_Cifar10(nn.Module):
    def __init__(self):
        super(Net_Cifar10, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(64, 10)

        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x):
        conv_ = self.conv(x)
        fc_ = conv_.view(-1, 8 * 8 * 32)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        return output


# Create clients and server
clients = []
for idx in range(args.num_clients):
    model = Net_Cifar10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # Sample data with uniform distribution
    # sampler = torch.utils.data.DistributedSampler(
    #     train_dataset, num_replicas=args.num_clients, rank=idx, shuffle=True
    # )

    # Sample data with Dirichlet distribution
    sampler = DirichletSampler(
        dataset=train_dataset, size=args.num_clients, rank=idx, alpha=args.alpha
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, **kwargs
    )
    device = f"cuda:{idx % torch.cuda.device_count()}" if args.cuda else "cpu"
    clients.append(
        Agent(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
        )
    )
server = Server(model=Net_Cifar10(), criterion=criterion)


def local_update_selected_clients(clients: list[Agent], server, local_update):
    train_loss_avg, train_acc_avg = 0, 0
    for client in clients:
        train_loss, train_acc = client.train_k_step(k=local_update)
        train_loss_avg += train_loss
        train_acc_avg += train_acc
    return train_loss_avg / len(clients), train_acc_avg / len(clients)


writer = SummaryWriter(
    os.path.join(
        "output",
        "cifar10",
        f"{args.sampling_type}+q_{args.q}+alpha_{args.alpha}+num_clients_{args.num_clients}_newpara",
        # datetime.now().strftime("%m_%d-%H-%M-%S"),
    )
)

with tqdm(total=args.rounds, desc=f"Training:") as t:
    for round in range(0, args.rounds):
        # Sample
        sampled_clients = client_sampling(
            server.determine_sampling(args.q, args.sampling_type), clients
        )
        # Train
        [client.pull_model_from_server(server) for client in sampled_clients]
        train_loss, train_acc = local_update_selected_clients(
            clients=sampled_clients, server=server, local_update=args.local_update
        )
        server.avg_clients(sampled_clients)
        # Decay the learning rate every M rounds
        if round % 500 == 0:
            [client.decay_lr_in_optimizer(gamma=0.5) for client in clients]

        # Evaluation and logging
        writer.add_scalar("Loss/train", train_loss, round)
        writer.add_scalar("Accuracy/train", train_acc, round)
        if round % 10 == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            writer.add_scalar("Loss/test", eval_loss, round)
            writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round+1}): {eval_loss=:.4f} {eval_acc=:.3f}")
            log(round, eval_acc)

        # Tqdm update
        t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
        t.update(1)


print(
    "Number of uniform participation rounds: " + str(server.get_num_uni_participation())
)
print(
    "Number of arbitrary participation rounds: "
    + str(server.get_num_arb_participation())
)
print("Ratio=" + str(server.get_num_arb_participation() / args.rounds))

eval_loss, eval_acc = server.eval(test_loader)
writer.add_scalar("Loss/test", eval_loss, round)
writer.add_scalar("Accuracy/test", eval_acc, round)
print(f"Evaluation(final round): {eval_loss=:.4f} {eval_acc=:.3f}")
writer.close()
