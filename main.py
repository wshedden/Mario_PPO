import argparse
from utils import get_device
from train import train_model, load_model_for_training
from gui import create_gui


def parse_args():
    # this function parses the command-line arguments and returns the arguments object
    # create an argument parser object
    parser = argparse.ArgumentParser(description="Train a PPO model on Super Mario Bros.")
    parser.add_argument("model", type=str, help="the name of the model to use or 'new' for a default model")
    parser.add_argument("-n", "--num_cpu", type=int, default=2, help="the number of CPUs to use (default: 2)")
    parser.add_argument("-s", "--skip", type=int, default=2, help="the number of frames to skip (default: 2)")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.000025, help="the learning rate for the model (default: 0.000025)")
    parser.add_argument("-d", "--log_dir", type=str, default="tmp/", help="the log directory for the model (default: tmp/)")
    parser.add_argument("-e", "--env_id", type=str, default="SuperMarioBros-1-1-v0", help="the environment id for the game (default: SuperMarioBros-1-1-v0)")
    return parser.parse_args()


if __name__ == "__main__":
    # parse the command-line arguments
    args = parse_args()

    # create the GUI
    create_gui(args)

    
