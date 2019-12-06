import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Get (sub)model of VGGFace model")
    parser.add_argument('--model', type=str, default='models/vggface.pth', help="input VGGFace model file")
    parser.add_argument('--output', type=str, default='models/vggface_conv.pth', help="output VGGFace (sub)model file")
    args = parser.parse_args()

    # Load model state dict
    model_state_dict = torch.load(args.model, map_location=lambda storage, loc: storage)

    # Old-to-new model state dict key map
    map_old2new_keys = {
        'features.conv_1_1.weight': 'conv_1_1.weight',
        'features.conv_1_1.bias': 'conv_1_1.bias',
        'features.conv_1_2.weight': 'conv_1_2.weight',
        'features.conv_1_2.bias': 'conv_1_2.bias',
        'features.conv_2_1.weight': 'conv_2_1.weight',
        'features.conv_2_1.bias': 'conv_2_1.bias',
        'features.conv_2_2.weight': 'conv_2_2.weight',
        'features.conv_2_2.bias': 'conv_2_2.bias',
        'features.conv_3_1.weight': 'conv_3_1.weight',
        'features.conv_3_1.bias': 'conv_3_1.bias',
        'features.conv_3_2.weight': 'conv_3_2.weight',
        'features.conv_3_2.bias': 'conv_3_2.bias',
        'features.conv_3_3.weight': 'conv_3_3.weight',
        'features.conv_3_3.bias': 'conv_3_3.bias',
        'features.conv_4_1.weight': 'conv_4_1.weight',
        'features.conv_4_1.bias': 'conv_4_1.bias',
        'features.conv_4_2.weight': 'conv_4_2.weight',
        'features.conv_4_2.bias': 'conv_4_2.bias',
        'features.conv_4_3.weight': 'conv_4_3.weight',
        'features.conv_4_3.bias': 'conv_4_3.bias',
        'features.conv_5_1.weight': 'conv_5_1.weight',
        'features.conv_5_1.bias': 'conv_5_1.bias',
        'features.conv_5_2.weight': 'conv_5_2.weight',
        'features.conv_5_2.bias': 'conv_5_2.bias',
        'features.conv_5_3.weight': 'conv_5_3.weight',
        'features.conv_5_3.bias': 'conv_5_3.bias'
    }

    new_model_state_dict = {}
    for old_key, new_key in map_old2new_keys.items():
        new_model_state_dict.update({new_key: model_state_dict[old_key]})

    # Save output model state dict
    torch.save(new_model_state_dict, args.output)
