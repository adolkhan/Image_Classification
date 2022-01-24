from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATASET_NAME_TO_DATASET_CLASS = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,

}


def create_dataloader(dataset_name):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = DATASET_NAME_TO_DATASET_CLASS[dataset_name](root="./", train=True,
                       download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128,
                             shuffle=True, num_workers=4)

    testset = DATASET_NAME_TO_DATASET_CLASS[dataset_name](root="./", train=False,
                       download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128,
                            shuffle=True, num_workers=4)
    return trainloader, testloader




