import numpy as np
import random
from shared.agent_utils import Agent
import functools


def client_sampling(
    sampling_type: str,
    clients: list[Agent],
    round: int,
    weights=None,
) -> list[Agent]:
    if sampling_type == "uniform":
        return uniform_client_sampling(clients)
    elif sampling_type == "afl":
        return afl_client_sampling(clients)
    elif sampling_type == "gamma":
        gamma_with_value = functools.partial(
            gamma_client_sampling, shape=10, scale=0.01
        )
        return gamma_with_value(clients)
    elif sampling_type == "beta":
        beta_with_value = functools.partial(beta_client_sampling, alpha=1, beta=10)
        return beta_with_value(clients)
    elif sampling_type == "markov":
        return markov_client_sampling(clients)
    elif sampling_type == "weibull":
        weibull_with_value = functools.partial(weibull_client_sampling, shape=10)
        return weibull_with_value(clients)
    elif sampling_type == "cyclic":
        return cyclic_client_sampling(clients, round)
    elif sampling_type == "circular":
        return circular_client_sampling(clients, round)
    else:
        raise Exception(f"Unsupported Sampling Type: {sampling_type}. ")


def afl_client_sampling(clients: list[Agent]) -> list[Agent]:
    num_to_draw = random.randint(1, len(clients))
    sampled_clients = random.sample(clients, num_to_draw)
    return sampled_clients


def uniform_client_sampling(clients: list[Agent]) -> list[Agent]:
    sampled_clients = random.sample(clients, int(len(clients) * 0.1))
    return sampled_clients


def gamma_client_sampling(
    clients: list[Agent], shape: float, scale: float
) -> list[Agent]:
    gamma_sample_indices = []
    while len(gamma_sample_indices) < len(clients) * 0.1:
        idx = int(np.random.gamma(shape=shape, scale=scale, size=1) * len(clients))
        if (idx not in gamma_sample_indices) and (idx < len(clients)):
            gamma_sample_indices.append(idx)
    sampled_clients = [clients[i] for i in gamma_sample_indices]
    return sampled_clients


def beta_client_sampling(
    clients: list[Agent], alpha: float, beta: float
) -> list[Agent]:
    beta_sample_indices = []
    while len(beta_sample_indices) < len(clients) * 0.1:
        idx = int(np.random.beta(alpha, beta, size=1) * len(clients))
        if idx not in beta_sample_indices:
            beta_sample_indices.append(idx)
    sampled_clients = [clients[i] for i in beta_sample_indices]
    return sampled_clients


def beta_client_sampling_with_weights(
    clients: list[Agent], alpha: float, beta: float, weights: list[float]
) -> tuple[list[Agent], list[float]]:
    beta_sample_indices = []
    beta_sample_weights = []
    while len(beta_sample_indices) < len(clients) * 0.1:
        idx = int(np.random.beta(alpha, beta, size=1) * len(clients))
        if idx not in beta_sample_indices:
            beta_sample_indices.append(idx)
            beta_sample_weights.append(idx)
    sampled_clients = [clients[i] for i in beta_sample_indices]
    sampled_weights = [weights[i] for i in beta_sample_weights]
    return sampled_clients, sampled_weights


def cyclic_client_sampling(clients: list[Agent], round: int) -> list[Agent]:
    num_groups = 4
    length_each_group = int(len(clients) / num_groups)
    start_index = int((round % num_groups) * length_each_group)
    sampled_clients = np.random.choice(
        clients[start_index : start_index + length_each_group],
        size=int(len(clients) * 0.1),
        replace=False,
    )
    return sampled_clients


def circular_client_sampling(clients: list[Agent], round: int) -> list[Agent]:
    num_groups = 10
    length_each_group = int(len(clients) / num_groups)
    start_index = int((round % num_groups) * length_each_group)
    end_index = start_index + length_each_group
    sampled_clients = clients[start_index:end_index]
    return sampled_clients


def weibull_client_sampling(clients: list[Agent], shape: float) -> list[Agent]:
    weibull_sample_indices = []
    while len(weibull_sample_indices) < len(clients) * 0.1:
        idx = int(np.random.weibull(a=shape, size=1) / 1.2 * len(clients))
        if (idx not in weibull_sample_indices) and (idx < len(clients)):
            weibull_sample_indices.append(idx)
    sampled_clients = [clients[i] for i in weibull_sample_indices]
    return sampled_clients


def markov_client_sampling(clients):
    return
