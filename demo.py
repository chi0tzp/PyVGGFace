import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lib import VGGFace


if __name__ == '__main__':
    # TODO: add comments
    model = VGGFace().double()
    model_dict = torch.load('models/vggface.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)

    # Load test image
    im = cv2.imread("data/ak.png")
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double()

    model.eval()
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    preds = F.softmax(model(im), dim=1)

    values, indices = preds.max(-1)
    print(indices)
