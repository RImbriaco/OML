import argparse
from src.factory import MethodFactory


parser = argparse.ArgumentParser()
parser.add_argument('--config_retrieval', type=str, default=None,
                    help='path to retrieval config file')
parser.add_argument('--config_rerank', type=str, default=None,
                    help='path to retrieval config file')


if __name__ == '__main__':
    args = parser.parse_args()
    method_factory = MethodFactory(args.config_retrieval,
                                   args.config_rerank)
    method_factory.deploy()









