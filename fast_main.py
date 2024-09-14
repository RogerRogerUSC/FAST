import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from shared.agent_utils import Agent, Server, DatasetSplit, data_each_node
from tqdm import tqdm
from shared.client_sampling import client_sampling
from shared.log import log
from models import cnn, lstm
from shared.agent_utils import local_update_selected_clients_fedavg
from config import get_parms
from fedlab.utils.dataset.partition import MNISTPartitioner, FMNISTPartitioner
import preprocess


args = get_parms("MNIST").parse_args()
torch.manual_seed(args.seed)
train_dataset, test_loader, device = preprocess.preprocess(args)

if args.dataset == "mnist":
    train_dataset_partition = MNISTPartitioner(
        targets=train_dataset.targets,
        num_clients=args.num_clients,
        partition="iid",  # "iid" if iid, "noniid-labeldir" if non-iid
        dir_alpha=None,  # "None" if iid, args.alpha if non-iid
        seed=args.seed,
    )
elif args.dataset == "fashion":
    train_dataset_partition = FMNISTPartitioner(
        targets=train_dataset.targets,
        num_clients=args.num_clients,
        partition="noniid-labeldir",
        # partition = "iid",
        dir_alpha=args.alpha,
        # dir_alpha=None,
        seed=args.seed,
    )
elif args.dataset == "cifar10":
    dict_users = data_each_node(train_dataset, args.num_clients, args.alpha)
elif args.dataset == "shakespeare":
    dict_users = train_dataset.get_client_dic()
    args.num_clients = len(dict_users)


# Create clients and server
clients = []
for idx in range(args.num_clients):
    if args.dataset == "mnist":
        model = cnn.CNN_Mnist().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.5,
        )
        scheduler = None
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(
                train_dataset, train_dataset_partition[idx]
            ),
            batch_size=args.train_batch_size,
        )
        server = Server(model=cnn.CNN_Mnist(), criterion=criterion, device=device)
    elif args.dataset == "fashion":
        model = cnn.CNN_FMNIST()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20000], gamma=0.1
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(
                train_dataset, train_dataset_partition[idx]
            ),
            batch_size=args.train_batch_size,
        )
        server = Server(model=cnn.CNN_FMNIST(), criterion=criterion, device=device)
    elif args.dataset == "cifar10":
        model = cnn.CNN_Cifar10_2().to(device)
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
        server = Server(model=cnn.CNN_Cifar10_2(), criterion=criterion, device=device)
    elif args.dataset == "shakespeare":
        model = lstm.CharLSTM()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20000], gamma=0.1
        )
        train_loader = torch.utils.data.DataLoader(
            DatasetSplit(train_dataset, dict_users[idx]),
            batch_size=args.train_batch_size,
        )
        server = Server(model=lstm.CharLSTM(), criterion=criterion, device=device)

    clients.append(
        Agent(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            scheduler=scheduler,
        )
    )


if args.log_to_tensorboard is not None:
    writer = SummaryWriter(
        os.path.join(
            "tensorboards",
            f"{args.dataset}",
            f"{args.log_to_tensorboard},{args.algo},local_update={args.local_update},nc={args.num_clients},rounds={args.round},lr={args.lr},seed={args.seed}",
        )
    )


list_q = []
v = 0
delta = 0
if args.adaptive == 1:
    q = 0
else:
    q = args.q

print(args)
with tqdm(total=args.round, desc=f"Training:") as t:
    for round in range(0, args.round):
        # Sample clients
        sampled_clients = client_sampling(
            server.determine_sampling(q, args.sampling_type),
            clients=clients,
            round=round,
        )
        # Training
        [client.pull_model_from_server(server) for client in sampled_clients]
        if args.algo in ["fedavg", "fedavgm"]:
            train_loss, train_acc = local_update_selected_clients_fedavg(
                clients=sampled_clients, server=server, local_update=args.local_update
            )
        server.avg_clients(sampled_clients, weights=None)
        # Evaluation and logging
        if args.log_to_tensorboard is not None:
            writer.add_scalar("Loss/train", train_loss, round)
            writer.add_scalar("Accuracy/train", train_acc, round)
        t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
        if (round + 1) % args.eval_iterations == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            if args.log_to_tensorboard is not None:
                writer.add_scalar("Loss/test", eval_loss, round)
                writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round}): {eval_loss=:.3f} {eval_acc=:.3f}")
            log(round, eval_acc)
        # Adaptive FAST
        if args.adaptive == 1:
            v = train_acc
            delta = delta - v
            q = min(1, max(0, q + args.gamma * delta))
            list_q.append({"Step": round, "Value": q})
            delta = v
        t.set_postfix({"loss": train_loss, "accuracy": 100.0 * train_acc})
        t.update(1)


eval_loss, eval_acc = server.eval(test_loader)
print(f"Evaluation(final round): {eval_loss=:.3f} {eval_acc=:.3f}")
if args.log_to_tensorboard is not None:
    writer.add_scalar("Loss/test", eval_loss, round)
    writer.add_scalar("Accuracy/test", eval_acc, round)
    writer.close()

# save the figure of how q changes
if args.adaptive == 1 and args.log_to_tensorboard is not None:
    fields = ["Step", "Value"]
    with open(
        f"results/q/{args.dataset}_q_{args.sampling_type},alpha={args.alpha},lambda={args.gamma}.csv",
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for data in list_q:
            writer.writerow(data)
