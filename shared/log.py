import logging
import os


def log(round, eval_acc, file_name="training.log"):
    filename = os.path.join("output", file_name)
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Epoch {round + 1}, Accuracy: {eval_acc=:.3f}")
