import os
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from agent_utils import Agent, Server
from tqdm import tqdm
from client_sampling import client_sampling
from log import log
from models.cnn import CNN_Mnist
from agent_utils import (
    local_update_selected_clients,
    local_update_selected_clients_fedprox,
)
from config import get_parms
from fedlab.utils.dataset.partition import MNISTPartitioner


args = get_parms("MNIST").parse_args()
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
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)

train_dataset_partition = MNISTPartitioner(
    targets=train_dataset.targets,
    num_clients=args.num_clients,
    partition="noniid-labeldir",
    dir_alpha=args.alpha,
    seed=args.seed,
)


# Create clients and server
clients = []
for idx in range(args.num_clients):
    model = CNN_Mnist().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.5,
    )
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(train_dataset, train_dataset_partition[idx]),
        batch_size=args.train_batch_size,
        **kwargs,
    )
    clients.append(
        Agent(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
        )
    )

server = Server(model=CNN_Mnist(), criterion=criterion, device=device)


writer = SummaryWriter(
    os.path.join(
        "output",
        "mnist",
        f"{args.sampling_type},alpha={args.alpha},q={args.q},{args.algo},ada={args.adaptive},nc={args.num_clients},rounds={args.rounds},lr={args.lr},",
    )
)


list_q = []
v = 0
delta = 0
gamma = 7
if args.adaptive == True:
    q = 0
else:
    q = args.q

with tqdm(total=args.rounds, desc=f"Training:") as t:
    for round in range(0, args.rounds):
        # Sample clients
        sampled_clients = client_sampling(
            server.determine_sampling(q, args.sampling_type),
            clients=clients,
            round=round,
        )
        # Train
        [client.pull_model_from_server(server) for client in sampled_clients]
        if args.algo == "fedavg":
            train_loss, train_acc = local_update_selected_clients(
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
        t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
        t.update(1)
        if round % 10 == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            writer.add_scalar("Loss/test", eval_loss, round)
            writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round}): {eval_loss=:.3f} {eval_acc=:.3f}")
            log(round, eval_acc)
        # Adaptive FAST
        if args.adaptive == True:
            v = train_acc
            delta = delta - v
            q = min(1, max(0, q + gamma * delta))
            list_q.append({"Step": round, "Value": q})
            delta = v


print(f"Number of uniform participation rounds: {server.get_num_uni_participation()}")
print(f"Number of arbitrary participation rounds: {server.get_num_arb_participation()}")
print(f"Ratio={server.get_num_arb_participation() / args.rounds}")

eval_loss, eval_acc = server.eval(test_loader)
writer.add_scalar("Loss/test", eval_loss, round)
writer.add_scalar("Accuracy/test", eval_acc, round)
print(f"Evaluation(final round): {eval_loss=:.3f} {eval_acc=:.3f}")
writer.close()

# save the figure of changing in q
if args.adaptive == True:
    fields = ["Step", "Value"]
    with open(
        f"results/q/list_fashion_q_{args.sampling_type},alpha={args.alpha},lambda={gamma}.csv",
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for data in list_q:
            writer.writerow(data)
