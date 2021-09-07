import os
import re
import torch
import pprint
from torch import nn


def get_device(gpu_id):
    """
    Allows for placing the model into 1 GPU and the data into other GPUs
     in case more than 1 is available.
    :param gpu_id: GPU to be used.
    :return:
    Dictionary specifying where the data and model will be assigned.
    """
    if len(gpu_id) == 1 and torch.cuda.device_count() >= 1:
        model_idx = 'cuda:' + str(gpu_id[0])
        unit_idx = 'cuda:' + str(gpu_id[0])
    elif len(gpu_id) > 1 and torch.cuda.device_count() > 1:
        model_idx = gpu_id  # list
        unit_idx = gpu_id[0]  # first device for loss etc.
    else:
        raise ValueError('CUDA device not found!')
    gpu_dict = {'model': model_idx, 'unit': unit_idx}
    return gpu_dict


def model_to_device(model, gpu_id):
    """
    Push model to single CUDA device.
    :param model: CNN to train/test.
    :param gpu_id: GPU to be used.
    :return:
    Model to corresponding CUDA device.
    """
    gpu_dict = get_device(gpu_id)
    if len(gpu_id) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_dict['model']).cuda()
    elif len(gpu_id) == 1 and torch.cuda.device_count() >= 1:
        model.to(gpu_dict['model'])
    else:
        raise ValueError('CUDA device not found!')
    return model



def get_latest_ckpt(dir_path):
    """
    Loads the latest checkpoint from the specified config file.
    :param dir_path: Path to the saved checkpoint folder
    :return:
    Full path of the last saved checkpoint.
    """
    latest_id = []
    for ckpt_name in os.listdir(dir_path):
        latest_id.append(re.match(
            re.compile('epoch_(\d+).pth'), ckpt_name).group(1))
    latest_id = max([int(x) for x in latest_id])
    return dir_path + os.sep + 'epoch_' + str(latest_id) + '.pth'


def check_and_make_dir(path):
    """
    Creates directory if missing.
    :param path: Path to save experiment.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_dirs(config):
    """
    Create the base paths for training.
    :param config: Retrieval config file.
    :return:
    Dictionary of target directories by name.
    """
    path_maps = {}
    base_dir = os.path.join(config['root_dir'], 'experiments')
    model_name_str = [config['train']['dataset'], config['train']['learning_rate'],
                      config['model']['backbone'], config['model']['pooling'],
                      config['model']['module'], config['experiment']]
    model_name_str = [f for f in model_name_str if f is not None]
    model_name = '_'.join(model_name_str)
    path_maps['model_name'] = model_name

    for dir_name in ['checkpoints', 'summary', 'outputs']:
        full_path = base_dir + os.sep + dir_name + os.sep
        check_and_make_dir(full_path)
        check_and_make_dir(full_path + os.sep + model_name)
        path_maps[dir_name] = full_path + os.sep + model_name
    return path_maps


def pretty_print(results):
    """
    Print evaluation results nicely.
    :param results: Result dictionary.
    """
    blacklist = ['ranks', 'qr_idx', 'db_idx']
    printable = {k: v for k, v in results.items() if k not in blacklist}
    pp = pprint.PrettyPrinter()
    pp.pprint(printable)
