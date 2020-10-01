import argparse


def get_dagan_args():
    parser = argparse.ArgumentParser(
        description="Welcome to GAN-Shot-Learning script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Filepath for dataset on which to train dagan. File should be .npy format with shape "
        "(num_classes, samples_per_class, height, width, channels).",
    )
    parser.add_argument(
        "final_model_path", type=str, help="Filepath to save final dagan model."
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="batch_size for experiment",
    )
    parser.add_argument(
        "--img_size",
        nargs="?",
        type=int,
        help="Dimension to scale images when training. "
        "Useful when model architecture expects specific input size. "
        "If not specified, uses img_size of data as passed.",
    )
    parser.add_argument(
        "--num_training_classes",
        nargs="?",
        type=int,
        default=1200,
        help="Number of classes to use for training.",
    )
    parser.add_argument(
        "--num_val_classes",
        nargs="?",
        type=int,
        default=200,
        help="Number of classes to use for validation.",
    )
    parser.add_argument(
        "--epochs",
        nargs="?",
        type=int,
        default=50,
        help="Number of epochs to run training.",
    )
    parser.add_argument(
        "--max_pixel_value",
        nargs="?",
        type=float,
        default=1.0,
        help="Range of values used to represent pixel values in input data. "
        "Assumes lower bound is 0 (i.e. range of values is [0, max_pixel_value]).",
    )
    parser.add_argument(
        "--save_checkpoint_path",
        nargs="?",
        type=str,
        help="Filepath to save intermediate training checkpoints.",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        nargs="?",
        type=str,
        help="Filepath of intermediate checkpoint from which to resume training.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        nargs="?",
        default=0.5,
        help="Dropout rate to use within network architecture.",
    )
    parser.add_argument(
        "--suppress_generations",
        action="store_true",
        help="If specified, does not show intermediate progress images.",
    )
    return parser.parse_args()
