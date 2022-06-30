from torchvision.models.resnet import ResNet, BasicBlock

class Resnet18(ResNet):
    def __init__(self):
        super(Resnet18, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.file_path = "data/resnet18.pth"

