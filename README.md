# FAST: Federated Average with Snapshot

FAST is a framework that can integrate almost all first-order Federated Learning algorithms. It provides multiple client participation patterns that we call arbitrary client participation. 

## Quick Start
### Run Experiments
1. **Run regular FAST**: `python fast_main.py --dataset=mnist --train-batch-size=128 --test-batch-size=256 --lr=0.01 --sampling-type=uniform_beta --local-update=5 --num-clients=100 --round=1000 --q=0.5 --alpha=0.1 --seed=365 --algo=fedavg --log-to-tensorboard=mnist01 --eval-iterations=10`
2. **Run adaptive FAST**: `python fast_main.py --dataset=mnist --train-batch-size=128 --test-batch-size=256 --lr=0.01 --sampling-type=uniform_beta --local-update=5 --num-clients=100 --round=1000 --q=0.5 --gamma=0.7 --alpha=0.1 --seed=365 --algo=fedavg --adaptive=1 --log-to-tensorboard=mnist01 --eval-iterations=10`
3. **Run regular FL**: `python fast_main.py --dataset=mnist --train-batch-size=128 --test-batch-size=256 --lr=0.01 --sampling-type=uniform --local-update=5 --num-clients=100 --round=1000 --seed=365 --algo=fedavg --log-to-tensorboard=mnist01 --eval-iterations=10`

## Performance


## Citation


## Contributors
This framework has been finished and maintained by **Zhe Li**(RIT), who was at the University of Southern California, **Bicheng Ying**(Google), and advised by Prof. **Haibo Yang**(RIT). 

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/b3982917-e302-42c3-b396-e33bb9f52c90" alt="Image 1" style="width: 80%;" />
</div>
