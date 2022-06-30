from torchvision.models.resnet import ResNet, Bottleneck

class Resnet152(ResNet):
    def __init__(self):
        super(Resnet152, self).__init__(Bottleneck, [3, 8, 36, 3])
        self.file_path = "data/resnet152.pth"

