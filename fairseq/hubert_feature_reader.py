# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import fairseq
import soundfile as sf
import torch.nn.functional as F


class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000, min_chunk=1600, device='cuda'):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.min_chunk = min_chunk
        self.device = device
        self.model.to(self.device)

    def read_audio(self, path, ref_len=None, channel_id=None):
        try:
            wav, sr = sf.read(path)
        except:
            return None
        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], \
                "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id-1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        if x is None:
            return None
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                if x_chunk.size(1) >= self.min_chunk:
                    feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer,
                    )
                    feat.append(feat_chunk)
                del x_chunk
        del x
        torch.cuda.empty_cache()
        return torch.cat(feat, 1).squeeze(0)