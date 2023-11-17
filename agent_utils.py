import torch
import torch.nn as nn
import random


def set_all_param_zero(model):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


def set_flatten_model_back(model, x_flattern):
    with torch.no_grad():
        start = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p_extract = x_flattern[start : (start + p.numel())]
            p.set_(p_extract.view(p.shape).clone())
            if p.grad is not None:
                p.grad.zero_()
            start += p.numel()


def get_flatten_model_param(model):
    with torch.no_grad():
        return torch.cat(
            [p.detach().view(-1) for p in model.parameters() if p.requires_grad]
        )


def get_flatten_model_grad(model):
    with torch.no_grad():
        return torch.cat(
            [p.grad.detach().view(-1) for p in model.parameters() if p.requires_grad]
        )


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().float().mean()


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.n = 0

    def update(self, val):
        if isinstance(val, torch.Tensor):
            self.sum += val.detach().cpu()
        else:
            self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


class Agent:
    def __init__(self, *, model, optimizer, criterion, train_loader, device="cpu"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_loss = Metric("train_loss")
        self.train_accuracy = Metric("train_accuracy")
        self.device = device
        self.batch_idx = 0
        self.epoch = 0
        self.data_generator = self.get_one_train_batch()

    def get_one_train_batch(self):
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            yield batch_idx, (inputs, targets)


    def reset_epoch(self):
        self.data_generator = self.get_one_train_batch()
        self.batch_idx = 0
        self.epoch += 1
        self.train_loss = Metric("train_loss")
        self.train_accuracy = Metric("train_accuracy")

    def pull_model_from_server(self, server):
        set_flatten_model_back(self.model, server.flatten_params)

    def decay_lr_in_optimizer(self, gamma):
        for g in self.optimizer.param_groups:
            g['lr'] *= gamma
            print("*******************************"+str(g['lr']))

    def train_k_step(self, k):
        self.model.train()
        for i in range(k):
            try:
                batch_idx, (inputs, targets) = next(self.data_generator)
            except StopIteration:
                loss, acc = self.train_loss.avg, self.train_accuracy.avg
                self.reset_epoch()
                return loss, acc
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #self.optimizer.zero_grad()
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.train_loss.update(loss.item())
            self.train_accuracy.update(accuracy(outputs, targets).item())
        return self.train_loss.avg, self.train_accuracy.avg

    def eval(self, test_dataloader) -> tuple[float, float]:
        self.model.eval()
        val_accuracy = Metric("val_accuracy")
        val_loss = Metric("val_loss")
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            val_accuracy.update(accuracy(outputs, targets).item())
            val_loss.update(self.criterion(outputs, targets).item())
        return val_loss.avg, val_accuracy.avg


class Server:
    def __init__(self, *, model, criterion, device="cpu"):
        self.model = model
        self.flatten_params = get_flatten_model_param(self.model)
        self.criterion = criterion
        self.device = device
        self.num_arb_participation = 0
        self.num_uni_participation = 0

    def avg_clients(self, clients: list[Agent]):
        self.flatten_params.zero_()
        for client in clients:
            self.flatten_params += get_flatten_model_param(client.model)
        self.flatten_params.div_(len(clients))
        set_flatten_model_back(self.model, self.flatten_params)

    def eval(self, test_dataloader) -> tuple[float, float]:
        self.model.eval()
        val_accuracy = Metric("val_accuracy")
        val_loss = Metric("val_loss")
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            val_accuracy.update(accuracy(outputs, targets).item())
            val_loss.update(self.criterion(outputs, targets).item())
        return val_loss.avg, val_accuracy.avg

    # Determine the sampling method by q
    def determine_sampling(self, q, sampling_type):
        if "_" in sampling_type:
            sampling_methods = sampling_type.split("_")
            if random.random() < q: 
                self.num_uni_participation += 1
                return "uniform"
            else:
                self.num_arb_participation += 1
                return sampling_methods[1]
        else: 
            return sampling_type

    def get_num_uni_participation(self): 
        return self.num_uni_participation
    
    def get_num_arb_participation(self):
        return self.num_arb_participation