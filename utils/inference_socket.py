from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import transforms
from PIL import Image
import io
import torch
import json
import torch.nn.functional as F
import numpy as np
from utils.resnet18 import Resnet18
from utils.resnet152 import Resnet152

torch.set_num_threads(1)

IMAGE_SIZE = 224


class Cpu:
    def __init__(self):
        self.topk = 1
        self.image_processing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.cpu_batches_processed = 0
        with open('./data/index_to_name.json') as json_file:
            self.mapping = json.load(json_file)

    def pre_process(self, batch):
        self.cpu_batches_processed += 1
        images = []
        for image in batch:
            image = Image.open(io.BytesIO(image))
            image = self.image_processing(image)
            images.append(image)
        result = np.stack(images)
        return result

    def post_process(self, data):
        ps = F.softmax(torch.tensor(data), dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return zip([self.mapping[str(cls[0])][1] for cls in classes], probs)


class Gpu:
    def __init__(self, model_name):
        self.gpu_batches_processed = 0
        if model_name == "resnet18":
            self.model = Resnet18()
        elif model_name == "resnet152":
            self.model = Resnet152()
        self.model.load_state_dict(torch.load(self.model.file_path))
        self.model.to("cuda")
        self.model.eval()

    def process(self, batch):
        batch = np.frombuffer(batch, dtype="float32")
        batch = batch.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        batch = torch.tensor(batch)
        self.gpu_batches_processed += 1
        result = self.model(batch.to("cuda")).cpu().detach().numpy()
        return result.tobytes()
