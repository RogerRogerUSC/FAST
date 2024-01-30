import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from agent_utils import Agent, Server, DatasetSplit, data_each_node
from tqdm import tqdm
from client_sampling import client_sampling
from log import log
from config import get_parms
from models.cnn import CNN_Cifar10_2
from agent_utils import (
    local_update_selected_clients_fedavg,
    local_update_selected_clients_fedprox,
)


args = get_parms("CIFAR10").parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()
torch.manual_seed(args.seed)

if use_cuda:
    device = torch.device("cuda")
    print("===cuda")
elif use_mps:
    device = torch.device("mps")
    print("===mps")
else:
    device = torch.device("cpu")
    print("===cpu")

kwargs = (
    {"num_workers": args.num_workers, "pin_memory": True, "shuffle": True}
    if use_cuda
    else {}
)
current_loc = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
    **kwargs,
)

# Create clients and server
dict_users = data_each_node(train_dataset, args.num_clients, args.alpha)
clients = []
for idx in range(args.num_clients):
    if use_cuda:
        gpu_idx = idx % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_idx}")

    model = CNN_Cifar10_2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5000], gamma=0.1
    )
    train_loader = torch.utils.data.DataLoader(
        DatasetSplit(train_dataset, dict_users[idx]),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    clients.append(
        Agent(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            scheduler=scheduler,
        )
    )
server = Server(model=CNN_Cifar10_2(), criterion=nn.CrossEntropyLoss(), device=device)


writer = SummaryWriter(
    os.path.join(
        "output",
        "cifar10",
        f"{args.sampling_type},alpha={args.alpha},q={args.q},{args.algo},ada={args.adaptive},nc={args.num_clients},rounds={args.rounds},lr={args.lr},{datetime}",
    )
)


list_q = []
v = 0
delta = 0
gamma = 7
if args.adaptive == 1:
    q = 0
else:
    q = args.q

with tqdm(total=args.rounds, desc=f"Training:") as t:
    for round in range(0, args.rounds):
        # Sample clients
        sampled_clients = client_sampling(
            server.determine_sampling(q, args.sampling_type),
            clients,
            round=round,
        )
        # Train
        [client.pull_model_from_server(server) for client in sampled_clients]
        if args.algo in ["fedavg", "fedavgm"]:
            train_loss, train_acc = local_update_selected_clients_fedavg(
                clients=sampled_clients, server=server, local_update=args.local_update
            )
        elif args.algo == "fedprox":
            train_loss, train_acc = local_update_selected_clients_fedprox(
                clients=sampled_clients, server=server, local_update=args.local_update
            )
        server.avg_clients(sampled_clients)
        # Evaluation and logging
        writer.add_scalar("Loss/train", train_loss, round)
        writer.add_scalar("Accuracy/train", train_acc, round)
        if round % 10 == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            writer.add_scalar("Loss/test", eval_loss, round)
            writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round}): {eval_loss=:.3f} {eval_acc=:.3f}")
            log(round, eval_acc)
            print(args)
            # Adaptive FAST
        if args.adaptive == 1:
            v = train_acc
            delta = delta - v
            q = min(1, max(0, q + gamma * delta))
            list_q.append({"Step": round, "Value": q})
            delta = v
        t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
        t.update(1)


print(f"Number of uniform participation rounds: {server.get_num_uni_participation()}")
print(f"Number of arbitrary participation rounds: {server.get_num_arb_participation()}")
print(f"Ratio={server.get_num_arb_participation() / args.rounds}")

eval_loss, eval_acc = server.eval(test_loader)
writer.add_scalar("Loss/test", eval_loss, round)
writer.add_scalar("Accuracy/test", eval_acc, round)
print(f"Evaluation(final round): {eval_loss=:.3f} {eval_acc=:.3f}")
writer.close()

# save the figure of changing in q
if args.adaptive == 1:
    fields = ["Step", "Value"]
    with open(
        f"results/q/list_cifar10_q_{args.sampling_type},alpha={args.alpha},lambda={gamma}.csv",
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for data in list_q:
            writer.writerow(data)
