import os
from datetime import datetime


def log_dir(output_path, prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = prefix + "run-" + now
    return "{}/{}/".format(output_path, name)


def train_dir(output_path):
  return os.path.join(output_path, 'train')


def eval_dir(output_path):
  return os.path.join(output_path, 'eval')


def model_dir(output_path):
  return os.path.join(output_path, 'model')
