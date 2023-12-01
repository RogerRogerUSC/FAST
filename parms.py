import argparse

# Parameters
def get_parms(dataset): 
    parser = argparse.ArgumentParser(description="PyTorch"+str(dataset)+"trainning")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--sampling_type", type=str, default="uniform_weibull", help="")
    parser.add_argument("--local_update", type=int, default=10, help="Local iterations")
    parser.add_argument(
        "--num_clients", type=int, default=100, help="Total number of clients"
    )
    parser.add_argument("--rounds", type=int, default=500, help="The number of rounds")
    parser.add_argument("--q", type=float, default=0.5, help="Probability q")
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Dirichlet Distribution parameter"
    )
    return parser