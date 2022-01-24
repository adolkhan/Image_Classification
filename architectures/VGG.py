import torch.nn as nn


vgg11_arch = [64, "MaxPool", 128, "MaxPool", 256, 256, "MaxPool", 512, 512, "MaxPool", 512, 512, "MaxPool"]
vgg13_arch = [64, 64, "MaxPool", 128, 128, "MaxPool", 256, 256, "MaxPool", 512, 512, "MaxPool", 512, 512, "MaxPool"]
vgg16_arch = [64, "Dropout", 64, "MaxPool", 128, "Dropout", 128, "MaxPool", 256,"Dropout", 256, "Dropout", 256,
              "MaxPool", 512, "Dropout", 512, "Dropout", 512, "MaxPool", 512, "Dropout", 512, "Dropout", 512,
              "Dropout", "MaxPool"]
vgg19_arch = [64, 64, "MaxPool", 128, 128, "MaxPool", 256, 256, 256, 256, "MaxPool", 512, 512, 512, 512, "MaxPool", 512,
              512, 512, 512, "MaxPool"]

cfg = {
    "vgg11": vgg11_arch,
    "vgg13": vgg13_arch,
    "vgg16": vgg16_arch,
    "vgg19": vgg19_arch,
}


class VGGBase(nn.Module):
    def __init__(self, vgg_model_name, cfg=cfg, dropout=0.5, num_classes=10, hidden=512):
        super().__init__()
        self.conv_layers = self.parse_cfg(cfg, vgg_model_name)
        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_head = self.make_head(dropout, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier_head(x)
        return x

    @staticmethod
    def make_head(dropout, num_classes, hidden=512):
        return nn.Sequential(*[
            nn.Linear(512 * 1 * 1, hidden),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes)
        ])

    @staticmethod
    def parse_cfg(cfg, vgg_model_name):
        layers = []
        in_channels = 3
        for layer in cfg[vgg_model_name]:
            if layer == "MaxPool":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif layer == "Dropout":
                layers += [nn.Dropout(p=0.4)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(layer)]
                in_channels = layer
        return nn.Sequential(*layers)
