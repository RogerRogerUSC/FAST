import torch
import numpy as np
import random


def client_sampling(sampling_type, clients, round):
    if sampling_type == "arbitrary":
        return arbitrary_client_sampling(clients)
    elif sampling_type == "uniform":
        return uniform_client_sampling(clients)
    elif sampling_type == "gamma":
        return gamma_client_sampling(clients)
    elif sampling_type == "bernoulli":
        return bernoulli_client_sampling(clients)
    elif sampling_type == "beta":
        return beta_client_sampling(clients)
    elif sampling_type == "markovian":
        return markovian_client_sampling(clients)
    elif sampling_type == "weibull":
        return weibull_client_sampling(clients)
    elif sampling_type == "cyclic":
        return cyclic_client_sampling(clients, round)
    else:
        raise Exception(f"Unsupported Sampling type: {sampling_type}. ")


# Mix all client sampling
def arbitrary_client_sampling(clients):
    random_number = random.random()
    threshold1 = 1 / 3
    threshold2 = 2 / 3
    if random_number < threshold1:
        return weibull_client_sampling(clients)
    elif random_number < threshold2:
        return cyclic_client_sampling(clients)
    else:
        return beta_client_sampling(clients)


def uniform_client_sampling(clients):
    uniform_samples_indices = np.random.uniform(high=len(clients), size=int(len(clients) * 0.1))
    uniform_samples_indices = uniform_samples_indices.astype(int)
    sampled_clients = [clients[i] for i in uniform_samples_indices]
    return sampled_clients


def gamma_client_sampling(clients):
    # Define parameters for the Gamma distribution
    gamma_shape = 3.0
    gamma_scale = 2.0
    # Generate weights using PyTorch's Gamma function
    weights = torch.tensor(
        np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=len(clients))
    )
    # Normalize weights to ensure their sum is 1
    normalized_weights = weights / weights.sum()
    # Sample clients using weights
    num_selected_clients = int(len(clients) * 0.1)
    selected_clients_indices = torch.multinomial(
        normalized_weights, num_selected_clients, replacement=False
    )
    # Return the selected clients
    selected_clients = [clients[i] for i in selected_clients_indices]
    return selected_clients


def beta_client_sampling(clients):
    alpha = 5
    beta = 1
    beta_samples_indices = np.random.beta(alpha, beta, size=int(len(clients) * 0.1))
    beta_samples_indices = (beta_samples_indices * len(clients)).astype(int)
    sampled_clients = [clients[i] for i in beta_samples_indices]
    return sampled_clients


# Cyclic client participation: Divide all clients into 5 groups.
def cyclic_client_sampling(clients, round):
    num_groups = 5
    length_each_group = int(len(clients) / num_groups)
    start_index = int((round % num_groups) * len(clients) / num_groups)
    sampled_clients = np.random.choice(
        clients[start_index : start_index + length_each_group],
        size=int(len(clients) * 0.1),
        replace=False,
    )
    return sampled_clients


def weibull_client_sampling(clients):
    shape = 1.0
    weibull_samples_indices = np.random.weibull(shape, size=int(len(clients) * 0.1))
    norm = np.linalg.norm(weibull_samples_indices)
    weibull_samples_indices = ((weibull_samples_indices / norm) * len(clients)).astype(
        int
    )
    sampled_clients = [clients[i] for i in weibull_samples_indices]
    return sampled_clients


def bernoulli_client_sampling(clients):
    sampled_clients = [client for client in clients if torch.rand(1).item() < 0.3][
        : int(len(clients) * 0.1)
    ]
    # sampled_clients = [client for client in clients if torch.rand(1).item()<0.3]
    return sampled_clients


def markovian_client_sampling(clients):
    transition_matrix = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    initial_distribution = torch.tensor([0.1, 0.9])
    current_state = torch.distributions.Categorical(initial_distribution).sample()
    sampled_clients = []
    for _ in range(int(len(clients) * 0.1)):
        # Generate the next state based on the current state
        next_state = torch.distributions.Categorical(
            transition_matrix[current_state]
        ).sample()
        # Append the current state as the sampled result
        sampled_clients.append(clients[next_state.item()])
        # Update the current state
        current_state = next_state
    return sampled_clients
