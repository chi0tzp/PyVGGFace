import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lib import VGGFace


if __name__ == '__main__':
    # Set up a parser for command line arguments
    parser = argparse.ArgumentParser("VGGFace demo script")
    parser.add_argument('--img', type=str, default='data/rm.jpg', help='input image file')
    # TODO: add CUDA acceleration
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use CUDA acceleration')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='do NOT use CUDA acceleration')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()

    # Get names list
    names = [line.rstrip('\n') for line in open('data/names.txt')]

    # Build VGGFace model and load pre-trained weights
    model = VGGFace().double()
    model_dict = torch.load('models/vggface.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)

    # Set model to evaluation mode
    model.eval()

    # Load test image and resize to 224x224
    img = cv2.imread(args.img)
    img = cv2.resize(img, (224, 224))

    # Forward test image through VGGFace
    img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224).double()
    img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    predictions = F.softmax(model(img), dim=1)
    score, index = predictions.max(-1)
    print("Predicted id: {} (probability: {})".format(names[index], score.item()))
