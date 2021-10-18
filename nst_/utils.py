import numpy as np
import torchvision.transforms as T
from PIL import Image


def load_img(path, size=512):

    image = Image.open(path)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(size),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return transform(image)[:3,:,:].unsqueeze(0)


def show_img(img_tensor):
    image = img_tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return T.ToPILImage()(T.ToTensor()(image.astype(np.float)))