import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        help='model_name resnet18 or resnet152')

    return parser.parse_args()