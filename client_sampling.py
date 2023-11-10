import torch
import numpy as np
import random


def client_sampling(sampling_type, clients):
    if sampling_type == "uniform":
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
        return cyclic_client_sampling(clients)


def uniform_client_sampling(clients):
    return random.sample(clients, int(len(clients) * 0.1))


def gamma_client_sampling(clients):
    # Define parameters for the Gamma distribution
    gamma_shape = 2.0
    gamma_scale = 1.0
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
    weights = torch.rand(len(clients))
    sampled_clients = [
        client
        for client, weight in zip(clients, weights)
        if weight < alpha / (alpha + beta)
    ][: int(len(clients) * 0.1)]
    return sampled_clients


def cyclic_client_sampling(clients):
    current_round = 1
    start_index = (current_round * int(len(clients) * 0.1)) % len(clients)
    sampled_clients = clients[start_index : start_index + int(len(clients) * 0.1)]
    return sampled_clients


def weibull_client_sampling(clients):
    shape = 1.5
    scale = 1.0
    # Generate random samples from the Weibull distribution
    samples = torch.distributions.Weibull(shape, scale).sample(
        sample_shape=(int(len(clients) * 0.1),)
    )
    # Map the samples to client indices
    sampled_clients_indices = (samples * len(clients)).long()
    # Ensure the selected indices are within bounds
    sampled_clients_indices = torch.clamp(sampled_clients_indices, 0, len(clients) - 1)
    # Get the selected clients
    sampled_clients = [clients[i] for i in sampled_clients_indices]
    return sampled_clients


def bernoulli_client_sampling(clients):
    sampled_clients = [client for client in clients if torch.rand(1).item() < 0.3][
        : int(len(clients) * 0.1)
    ]
    # sampled_clients = [client for client in clients if torch.rand(1).item()<0.3]
    return sampled_clients


def markovian_client_sampling(clients):
    transition_matrix = torch.tensor(
        [[0.2, 0.3, 0.5], [0.4, 0.2, 0.4], [0.1, 0.7, 0.2]]
    )
    initial_distribution = torch.tensor([0.3, 0.4, 0.3])
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
