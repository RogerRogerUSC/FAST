import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from agent_utils import Agent, Server
from tqdm import tqdm
from client_sampling import client_sampling
from log import log
from data_dist import get_data_sampler
from config import get_parms
from models.cnn import CNN_Cifar10
from models.resnet import ResNet34, ResNet18
from local_update import local_update_selected_clients
from fedlab.models.cnn import CNN_CIFAR10, AlexNet_CIFAR10
from FedLab.fedlab.contrib.dataset.partitioned_cifar10 import CIFAR10Partitioner


args = get_parms("CIFAR10").parse_args()
print(args)
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

kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True} if use_cuda else {}
current_loc = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    shuffle=False,
    **kwargs,
)

train_dataset_part = CIFAR10Partitioner(
    targets=train_dataset.targets,
    num_clients=args.num_clients,
    balance=True,
    partition="dirichlet",
    dir_alpha=args.alpha,
    seed=args.seed,
    verbose=False
)

# Create clients and server
clients = []
for idx in range(args.num_clients):
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
    )
    # optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(params=model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(train_dataset, train_dataset_part[idx]),
        batch_size=args.train_batch_size,
        **kwargs,
    )
    clients.append(
        Agent(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
        )
    )
server = Server(model=ResNet18(), criterion=criterion, device=device)


writer = SummaryWriter(
    os.path.join(
        "output", "cifar_test", f"{args}+{datetime.now().strftime('%m_%d-%H-%M')}"
    )
)

with tqdm(total=args.rounds, desc=f"Training:") as t:
    for round in range(0, args.rounds):
        # Sample clients
        sampled_clients = client_sampling(
            server.determine_sampling(args.q, args.sampling_type),
            clients,
            round=round,
        )
        # Train
        [client.pull_model_from_server(server) for client in sampled_clients]
        train_loss, train_acc = local_update_selected_clients(
            clients=sampled_clients, server=server, local_update=args.local_update
        )
        server.avg_clients(sampled_clients)
        # Decay the learning rate every M rounds
        if (round+1) % 500 == 0:
            [client.decay_lr_in_optimizer(gamma=0.5) for client in clients]

        # Evaluation and logging
        writer.add_scalar("Loss/train", train_loss, round)
        writer.add_scalar("Accuracy/train", train_acc, round)
        if round % 10 == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            writer.add_scalar("Loss/test", eval_loss, round)
            writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round}): {eval_loss=:.3f} {eval_acc=:.3f}")
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
print(f"Evaluation(final round): {eval_loss=:.3f} {eval_acc=:.3f}")
writer.close()
