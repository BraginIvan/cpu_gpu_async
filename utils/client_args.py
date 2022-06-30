import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports',
                        type=int,
                        help='number of ports to use')

    parser.add_argument('--batch-size',
                        default=24,
                        type=int)

    parser.add_argument('--imgs-path',
                        type=str)

    parser.add_argument('--images-n',
                        default=1608,
                        type=int)


    return parser.parse_args()