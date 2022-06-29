from torchvision import transforms
from PIL import Image
import io
import torch
import json
import torch.nn.functional as F
from architectures.model import ImageClassifier
torch.set_num_threads(1)



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
        with open('../data/index_to_name.json') as json_file:
            self.mapping = json.load(json_file)

    def pre_process(self, batch):
        self.cpu_batches_processed += 1
        images = []
        for image in batch:
            image = Image.open(io.BytesIO(image))
            image = self.image_processing(image)
            images.append(image)
        return torch.stack(images)

    def post_process(self, data):
        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return zip([self.mapping[str(cls[0])][1] for cls in classes], probs)


class Gpu:
    def __init__(self):
        self.gpu_batches_processed = 0
        self.model = ImageClassifier()
        m = torch.load("data/resnet18.pth")
        self.model.load_state_dict(m)
        self.model.to("cuda")
        self.model.eval()

    def process(self, batch):
        print("gpu")
        self.gpu_batches_processed += 1
        result = self.model(batch.to("cuda")).cpu().detach()
        return result
