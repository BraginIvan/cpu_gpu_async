import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        help='model_name resnet18 or resnet152')

    parser.add_argument('--ports',
                        default=1,
                        type=int,
                        help='number of ports to use')

    return parser.parse_args()