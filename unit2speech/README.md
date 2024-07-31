Environment:
```bash
git clone https://github.com/facebookresearch/speech-resynthesis.git
cd speech-resynthesis
pip install amfm_decompy
cp generate_waveform.py ./speech-resynthesis/
cd speech-resynthesis
```

Inference:
input file sould be in the format of jsonl file:
```
{"code": [1 1 1..., 2 1 3..., ...] or 1 1 1 ..., "name": <file_name>}
```
If code is of type list, it will interchange the two speakers.

```bash
# assign speaker id for spk1 and spk2 (represent the voice of speaker for User and Machine, 0 ~ 200)
python generate_waveform.py \
    --in-file <input_path> \
    --vocoder <vocoder_path> \
    --vocoder-cfg <vocoder_config_path> \
    --results-path <output_path> \
    --spk1 4 \
    --spk2 5
```
