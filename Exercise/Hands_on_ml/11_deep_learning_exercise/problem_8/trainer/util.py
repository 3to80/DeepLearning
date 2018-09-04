import os
from datetime import datetime


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "mnist_log"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def train_dir(output_path):
  return os.path.join(output_path, 'train')


def eval_dir(output_path):
  return os.path.join(output_path, 'eval')


def model_dir(output_path):
  return os.path.join(output_path, 'model')
