import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agent_utils import Agent, Server
from tqdm import tqdm
from client_sampling import client_sampling
from log import log
from agent_utils import (
    local_update_selected_clients_fedavg,
    local_update_selected_clients_fedprox,
    DatasetSplit,
)
from config import get_parms
from dataset import ShakeSpeare
from models.lstm import CharLSTM


args = get_parms("Shakespeare").parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if use_cuda:
    device = torch.device("cuda")
    print("------ Using cuda ------")
elif use_mps:
    device = torch.device("mps")
    print("------ Using mps ------")
else:
    device = torch.device("cpu")
    print("------ Using cpu ------")


kwargs = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {}
current_loc = os.path.dirname(os.path.abspath(__file__))
# pdataset = PickleDataset(
#     pickle_root=f"{current_loc}/FedLab/datasets/pickle_datasets",
#     data_root=f"{current_loc}/FedLab/datasets/datasets",
#     dataset_name="shakespeare",
# )
# # read saved pickle dataset and get responding dataset
# train_dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id=0)
# test_dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id=2)

train_dataset = ShakeSpeare(train=True)
test_dataset = ShakeSpeare(train=False)
dict_users = train_dataset.get_client_dic()
args.num_clients = len(dict_users)


test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)

# Create clients and server
clients = []
for idx in range(args.num_clients):
    model = CharLSTM()
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
        **kwargs,
    )
    clients.append(
        Agent(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            scheduler=scheduler,
            device=device,
        )
    )
server = Server(model=CharLSTM(), criterion=nn.CrossEntropyLoss(), device=device)


writer = SummaryWriter(
    os.path.join(
        "output",
        "shakespeare",
        f"{args.sampling_type},alpha={args.alpha},q={args.q},{args.algo},ada={args.adaptive},nc={args.num_clients},rounds={args.rounds},lr={args.lr},",
    )
)


list_q = []
v = 0
delta = 0
gamma = 1
if args.adaptive == 1:
    q = 0
else:
    q = args.q

with tqdm(total=args.rounds, desc=f"Training:") as t:
    for round in range(0, args.rounds):
        sampled_clients = client_sampling(
            server.determine_sampling(q, args.sampling_type), clients, round
        )
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
        writer.add_scalar("Loss/train", train_loss, round)
        writer.add_scalar("Accuracy/train", train_acc, round)
        # Evaluate once per 10 rounds
        if round % 10 == 0:
            eval_loss, eval_acc = server.eval(test_loader)
            writer.add_scalar("Loss/test", eval_loss, round)
            writer.add_scalar("Accuracy/test", eval_acc, round)
            print(f"Evaluation(round {round+1}): {eval_loss=:.4f} {eval_acc=:.3f}")
            log(round, eval_acc)
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
print(f"Evaluation(final round): {eval_loss=:.4f} {eval_acc=:.3f}")
writer.close()

# save the figure of changing in q
if args.adaptive == 1:
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
