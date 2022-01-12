from architectures.VGG import VGGBase

MODEL_NAME_TO_MODEL_CLASS = {
    "vgg": VGGBase,
}


def create_model(model, arch_type):
    return MODEL_NAME_TO_MODEL_CLASS[model](arch_type)


