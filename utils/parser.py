import argparse


def get_dagan_args():
    parser = argparse.ArgumentParser(
        description="Use this script to train a dagan.",
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


def get_omniglot_classifier_args():
    parser = argparse.ArgumentParser(
        description="Use this script to train an omniglot classifier "
        "with and without augmentations to compare the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "generator_path",
        type=str,
        help="Filepath for dagan generator to use for augmentations.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/omniglot_data.npy",
        help="Filepath for omniglot data.",
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="batch_size for experiment",
    )
    parser.add_argument(
        "--data_start_index",
        nargs="?",
        type=int,
        default=1420,
        help="Only uses classes after the given index for training. "
        "Useful to isolate data that wasn't used during dagan training.",
    )
    parser.add_argument(
        "--num_training_classes",
        nargs="?",
        type=int,
        default=100,
        help="Number of classes to use for training.",
    )
    parser.add_argument(
        "--train_samples_per_class",
        nargs="?",
        type=int,
        default=5,
        help="Number of samples to use per class during training.",
    )
    parser.add_argument(
        "--val_samples_per_class",
        nargs="?",
        type=int,
        default=5,
        help="Number of samples to use per class during validation.",
    )
    parser.add_argument(
        "--epochs",
        nargs="?",
        type=int,
        default=200,
        help="Number of epochs to run training.",
    )
    parser.add_argument(
        "--progress_frequency",
        nargs="?",
        type=int,
        default=50,
        help="Number of epochs between printing intermediate train/val loss.",
    )
    parser.add_argument(
        "--generated_batches_per_real",
        nargs="?",
        type=int,
        default=1,
        help="Number of augmented batches per real batch during "
        "augmented training phase.",
    )
    parser.add_argument(
        "--num_bootstrap_samples",
        nargs="?",
        type=int,
        default=10,
        help="Number of classifiers to train with slightly different data "
        "in order to get a more accuracy measurement.",
    )
    return parser.parse_args()
