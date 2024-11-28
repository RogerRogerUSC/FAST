import argparse


# Parameters
def get_parms(dataset):
    parser = argparse.ArgumentParser(description="PyTorch" + str(dataset) + "trainning")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sampling-type", type=str, default="uniform")
    parser.add_argument("--local-update", type=int, default=10, help="Local iterations")
    parser.add_argument("--num-clients", type=int, default=100, help="Total clients")
    parser.add_argument("--round", type=int, default=1000, help="Communication rounds")
    parser.add_argument("--q", type=float, default=1, help="Probability of snapshot")
    parser.add_argument(
        "--gamma", type=float, default=0.7, help="parameter for adaptive FAST"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet parameter")
    parser.add_argument(
        "--data-dist", type=str, default="dirichlet", help="[uniform, dirichlet]"
    )
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=365, help="random seed")
    parser.add_argument("--algo", type=str, default="fedavg", help="[fedavg, fedcom, fedavgm, fedprox, fedau]")
    parser.add_argument("--adaptive", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--compressor", type=str, default=None, help="[topk,randk, quan]"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="[mnist, fashion, cifar10, shakespeare]",
    )
    parser.add_argument("--log-to-tensorboard", type=str, default=None)
    parser.add_argument("--eval-iterations", type=int, default=10)
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    return parser
