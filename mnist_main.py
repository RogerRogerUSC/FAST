import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from agent_utils import Agent, Server
from tqdm import tqdm
from data_dist import GammaDataSampler
from client_sampling import client_sampling
from log import log

parser = argparse.ArgumentParser(description="PyTorch MNIST trainning")
parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
parser.add_argument("--epoch", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)",)
parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--sampling_type", type=str, default="uniform", help="")
parser.add_argument("--local_update", type=int, default=10, help="Local iterations")
parser.add_argument("--client_num", type=int, default=20, help="Total number of clients")
parser.add_argument("--round", type=int, default=30, help="Round number")
parser.add_argument("--q", type=float, default=0.5, help="Probability q")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
current_loc = os.path.dirname(os.path.abspath(__file__))
train_dataset = datasets.MNIST(
    os.path.join(current_loc, "data", "mnist"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_dataset = datasets.MNIST(
    os.path.join(current_loc, "data", "mnist"),
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, **kwargs
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


# Create clients and server
clients = []
for idx in range(args.client_num):
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # Sample data as uniform data distribution 
    sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=args.client_num, rank=idx, shuffle=True
    ) 
    #sampler = GammaDataSampler(train_dataset, args.client_num, 2.0, 1.0)
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
server = Server(model=Net(), criterion=criterion)


def local_update_selected_clients(clients: list[Agent], server, local_update):
    train_loss_avg, train_acc_avg = 0, 0
    for client in clients:
        train_loss, train_acc = client.train_k_step(k=local_update)
        train_loss_avg += train_loss / len(clients)
        train_acc_avg += train_acc / len(clients)
    return train_loss_avg, train_acc_avg


# for epoch in range(args.epoch):
#     with tqdm(total=len(clients[0].train_loader), desc=f"Training(epoch {epoch+1})") as t:
#         for _ in range(math.ceil(len(train_loader) / args.local_update)):
#             sampled_clients = client_sampling(args.sampling_type, clients)
#             [client.pull_model_from_server(server) for client in sampled_clients]
#             train_loss, train_acc = local_update_selected_clients(
#                 clients=sampled_clients, server=server, local_update=args.local_update
#             )
#             server.avg_clients(sampled_clients)

#             t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
#             t.update(args.local_update)
#     eval_loss, eval_acc = server.eval(test_loader)
#     print(f"Evaluation(epoch {epoch+1}): {eval_loss=:.4f} {eval_acc=:.3f}")

if os.path.exists("training.log"): os.remove("training.log")
for round in range(args.round):
    with tqdm(total=len(clients[0].train_loader), desc=f"Training(round {round+1})") as t:
        for _ in range(math.ceil(len(train_loader) / args.local_update)):
            sampled_clients = client_sampling(server.determine_sampling(args.q, args.sampling_type), clients)
            [client.pull_model_from_server(server) for client in sampled_clients]
            train_loss, train_acc = local_update_selected_clients(
                clients=sampled_clients, server=server, local_update=args.local_update
            )
            server.avg_clients(sampled_clients)
            t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
            t.update(args.local_update)
    eval_loss, eval_acc = server.eval(test_loader)
    print(f"Evaluation(round {round+1}): {eval_loss=:.4f} {eval_acc=:.3f}")
    # eval_acc_list = []
    # eval_acc_list.append(eval_acc)
    log(round, eval_acc)
