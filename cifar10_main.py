import os
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
from data_dist import get_data_sampler
from config import get_parms
from models.cnn_cifar10 import CNNCifar10, CNNCifar10_test
from local_update import local_update_selected_clients


args = get_parms("CIFAR10").parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
current_loc = os.path.dirname(os.path.abspath(__file__))

transform_train = transforms.Compose(
    [
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

# img_size = train_dataset[0][0].shape

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


# Create clients and server
clients = []
for idx in range(args.num_clients):
    model = CNNCifar10()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    sampler = get_data_sampler(dataset=train_dataset, args=args, idx=idx)
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
device = f"cuda:0" if args.cuda else "cpu"
server = Server(model=CNNCifar10_test(), criterion=criterion, device=device)


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
        # if round == 350:
        #     [client.decay_lr_in_optimizer(gamma=0.5) for client in clients]

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
