from agent_utils import Agent


def local_update_selected_clients(clients: list[Agent], server, local_update):
    train_loss_sum, train_acc_sum = 0, 0
    for client in clients:
        train_loss, train_acc = client.train_k_step(k=local_update)
        train_loss_sum += train_loss
        train_acc_sum += train_acc
    return train_loss_sum / len(clients), train_acc_sum / len(clients)
