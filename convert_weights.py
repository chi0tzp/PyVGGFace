import os
import os.path as osp
import torch
import torchfile
import tarfile
import urllib.request
from lib import VGGFace


def download_torch_weights(output_dir):
    torch_tar_file = osp.join(output_dir, 'vgg_face_torch.tar.gz')
    torch_weights_file = osp.join(output_dir, 'vgg_face_torch/VGG_FACE.t7')

    # Download tar.gz file
    if not osp.isfile(torch_tar_file):
        urllib.request.urlretrieve(url='http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz',
                                   filename=torch_tar_file)
    # Extract 'vgg_face_torch/VGG_FACE.t7' tar.gz file into 'models/vgg_face_torch/VGG_FACE.t7'
    if not osp.isfile(torch_weights_file):
        tar_file = tarfile.open(torch_tar_file)
        tar_file.extract(member='vgg_face_torch/VGG_FACE.t7', path=output_dir)
    return torch_weights_file


def convert(torch_weights_file, model):
    """ Convert LuaTorch weights and load them to PyTorch model

    Args:
        torch_weights_file (str) : filename of pre-trained LuaTorch weights file
        model (VGGFace)          : VGGFace model
    """
    torch_model = torchfile.load(torch_weights_file)
    counter = 1
    block = 1
    block_size = [2, 2, 3, 3, 3]
    for i, layer in enumerate(torch_model.modules):
        if layer.weight is not None:
            if block <= 5:
                self_layer = model.features['conv_{}_{}'.format(block, counter)]
                counter += 1
                if counter > block_size[block - 1]:
                    counter = 1
                    block += 1
                self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
            else:
                self_layer = model.fc['fc{}'.format(block)]
                block += 1
                self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
    return model


if __name__ == '__main__':
    # Create output directory
    output_dir = 'models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("#. Download and extract original pre-trained LuaTorch tar.gz weights file...")
    torch_weights_file = download_torch_weights(output_dir)

    # Define VGGFace instance
    print("#. Convert original pre-trained LuaTorch and load them to VGGFace model...")
    vggface_model = VGGFace()
    vggface_model = convert(torch_weights_file=osp.join(output_dir, 'vgg_face_torch/VGG_FACE.t7'),
                            model=vggface_model)

    # Save output model
    vggface_model_file = osp.join(output_dir, 'vggface.pth')
    print("#. Save VGGFace weights at {}".format(vggface_model_file))
    torch.save(vggface_model.state_dict(), vggface_model_file)
