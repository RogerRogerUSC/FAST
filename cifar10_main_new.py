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
from models.cnn_cifar10 import CNNCifar10
from local_update import local_update_selected_clients


# Parameters
parser = argparse.ArgumentParser(description="PyTorch MNIST trainning")
parser.add_argument("--batch-size",type=int,default=64,metavar="N",help="input batch size for training (default: 64)",)
parser.add_argument("--test-batch-size",type=int,default=256,metavar="N",help="input batch size for testing (default: 1000)",)
parser.add_argument( "--epochs",type=int,default=50,metavar="N",help="number of epochs to train (default: 10)",)
parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--sampling_type", type=str, default="uniform_weibull", help="")
parser.add_argument("--local_update", type=int, default=10, help="Local iterations")
parser.add_argument("--num_clients", type=int, default=250, help="Total number of clients")
parser.add_argument("--rounds", type=int, default=3000, help="The number of rounds")
parser.add_argument("--q", type=float, default=0.5, help="Probability q")
parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet Distribution parameter")
parser.add_argument("--num_channels", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=10)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
current_loc = os.path.dirname(os.path.abspath(__file__))
train_dataset = datasets.CIFAR10(
    os.path.join(current_loc, "data", "cifar10"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_dataset = datasets.CIFAR10(
    os.path.join(current_loc, "data", "cifar10"),
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)


model_global = CNNCifar10(args)

# Create clients and server
clients = []
for idx in range(args.num_clients):
    model = model_global
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    sampler = DirichletSampler(
        dataset=train_dataset, size=args.num_clients, rank=idx, alpha=args.alpha
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, **kwargs
    )
    clients.append(
        Agent(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
        )
    )
server = Server(model=CNNCifar10(args), criterion=criterion)


writer = SummaryWriter(
    os.path.join(
        "output",
        "cifar10",
        f"{args.sampling_type}+q_{args.q}+alpha_{args.alpha}+num_clients_{args.num_clients}",
        # datetime.now().strftime("%m_%d-%H-%M-%S"),
    )
)


with tqdm(total=args.rounds, desc=f"Training:") as t:
    for round in range(args.rounds):
        # Sample clients
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
        if round % 500 == 0 and round != 0:
            [client.decay_lr_in_optimizer(gamma=0.5) for client in clients]
        # Evaluation and logging
        writer.add_scalar("Loss/train", train_loss, round)
        writer.add_scalar("Accuracy/train", train_acc, round)
        if round % 10 == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            writer.add_scalar("Loss/test", eval_loss, round)
            writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round}): {eval_loss=:.4f} {eval_acc=:.3f}")
        # Tqdm update
        t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
        t.update(1)


print("Number of uniform participation rounds: " + str(server.get_num_uni_participation()))
print("Number of arbitrary participation rounds: " + str(server.get_num_arb_participation()))
print("Ratio=" + str(server.get_num_arb_participation() / args.rounds))

eval_loss, eval_acc = server.eval(test_loader)
writer.add_scalar("Loss/test", eval_loss, round)
writer.add_scalar("Accuracy/test", eval_acc, round)
print(f"Evaluation(final round): {eval_loss=:.4f} {eval_acc=:.3f}")
writer.close()
