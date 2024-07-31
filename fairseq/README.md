This folder is to encode speech into hubert tokens, which will be 2 to 3 times faster than the original fairseq implementation.
Data manifest:
```bash
python manifest.py <audio_root> --valid-percent 0 --dest <output_path> --ext flac --min-dur 0
```

Speech to unit:
```bash
python quantize_with_kmeans.py \
    --acoustic_model_path <hubert_path> \
    --layer 12 \
    --kmeans_model_path <kmeans_path> \
    --manifest_path <manifest_file_path> \
    --out_quantized_file_path <output_path> \
    --num-gpus 1
```
