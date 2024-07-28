# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import numpy as np
import torch
import joblib
from joblib import Parallel, delayed
import tqdm

from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    get_audio_files
)
from hubert_feature_reader import HubertFeatureReader
def gen_kmeans(
    checkpoint_path, layer, manifest_path, sample_pct, channel_id, kmeans_models, args
):
    def get_token_string(reader, kmeans_model, path):
        base_fname = os.path.basename(path).rstrip('.'+args.extension.lstrip('.'))
        feats = reader.get_feats(root_dir + '/' + path, channel_id=channel_id)
        if feats is not None:
            pred = torch.argmin(torch.cdist(feats, kmeans_model), -1).cpu().tolist()
            del feats
            torch.cuda.empty_cache()
            pred_str = " ".join(str(p) for p in pred)
        else:
            pred_str = ""

        if args.channel_id is not None:
            base_fname = base_fname+f'-channel{args.channel_id}'
        if not args.hide_fname:
            return f"{base_fname}|{pred_str}\n"
        else:
            return f"{pred_str}\n"

    readers = [HubertFeatureReader(checkpoint_path=checkpoint_path, layer=layer, device=f"cuda:{i}") for i in range(args.num_gpus)]

    root_dir, fnames, _ = get_audio_files(args.manifest_path)
    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    print(f"Writing quantized predictions to {args.out_quantized_file_path}")

    with open(args.out_quantized_file_path, "w") as fout:
        for line in Parallel(n_jobs=16, backend="threading", return_as="generator")(delayed(get_token_string)(readers[i % args.num_gpus], kmeans_models[i % args.num_gpus], path) for i, path in enumerate(tqdm.tqdm(fnames))):
            fout.write(line)


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    parser.add_argument(
        "--channel_id",
        choices=['1', '2'],
        help="The audio channel to extract the units in case of stereo file.",
        default=None,
    )
    parser.add_argument(
        "--hide-fname", action='store_true',
        help="Hide file names in the output file."
    )
    parser.add_argument(
        "--num-gpus", type=int,
        default=1,
        help="Number of GPUs"
    )
    return parser


def main(args, logger):
    
    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_models = [torch.from_numpy(kmeans_model.cluster_centers_).to(f"cuda:{i}") for i in range(args.num_gpus)]
    # kmeans_model.verbose = False
    
    # Feature extraction
    logger.info(f"Extracting Hubert acoustic features...")
    gen_kmeans(
        checkpoint_path=args.acoustic_model_path,
        layer=args.layer,
        manifest_path=args.manifest_path,
        sample_pct=1.0,
        channel_id=int(args.channel_id) if args.channel_id else None,
        kmeans_models=kmeans_models,
        args=args
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)