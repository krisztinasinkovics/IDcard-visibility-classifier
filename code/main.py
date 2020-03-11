from __future__ import absolute_import
import os
import argparse

from lib.train import train_model
from lib.predict import predict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Execution of the CV ID challenge task.')
    parser.add_argument('--mode', type=str, required=True, choices=('train','predict'),
                    help='Whether train or predict module should be executed')
    parser.add_argument('--img_dir', type=str, help='Directory where images are present.')
    parser.add_argument("--metadata_dir", type=str, help="Directory where gicsd_labels.csv is present.")
    parser.add_argument('--out_dir', type=str, help='output directory where model hdf5 file will be saved')
    parser.add_argument('--path_to_image', type=str, help='path to the image to delover predictions for')
    parser.add_argument('--model_location', type=str, help='path to the model hdf5 file')

    args = parser.parse_args()

    if args.mode == 'train' and (args.img_dir is None or args.metadata_dir is None or args.out_dir is None):
        parser.error("--train mode requires --img_dir and --metadata_dir.")

    if args.mode == 'predict' and (args.path_to_image is None or args.model_location is None):
        parser.error("--predict mode requires --path_to_image.")

    img_dir = "/kaggle/input/images-classification/data/images"
    metadata_dir = "/kaggle/input/images-classification/data/"
    out_dir = "/data/workspace/"

    if args.mode == 'train':
        train_model(args.img_dir, args.metadata_dir, args.out_dir)

    if args.mode == 'predict':
        predict(args.path_to_image, args.model_location)
