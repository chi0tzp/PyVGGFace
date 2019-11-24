#!/usr/bin/env bash

echo "Download original pre-trained LuaTorch tar.gz weights file..."
wget --no-clobber http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz -P models/

echo "Extract tar.gz file..."
tar -C models/ -xvf models/vgg_face_torch.tar.gz vgg_face_torch/VGG_FACE.t7
