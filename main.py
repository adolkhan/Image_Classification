import argparse

from trainer import Trainer
from utils import set_determenistic


def main(parser):
    args = parser.parse_args()
    set_determenistic()

    trainer = Trainer(
        args.dataset_name,
        args.model_name,
        args.architecture_type,
        args.optimizer_name,
        args.loss_name,
        args.project_name,
        args.epochs,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="CIFAR10")
    parser.add_argument("--model-name", type=str, default="vgg")
    parser.add_argument("--architecture-type", type=str, default="vgg16")
    parser.add_argument("--optimizer-name", type=str, default="Adam")
    parser.add_argument("--loss-name", type=str, default="CrossEntropy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--project-name", type=str, default="VGG CIFAR10")
    parser.add_argument("--hidden", type=int, default=512)
    main(parser)
