Install pyannote according to https://github.com/pyannote/pyannote-audio?tab=readme-ov-file.
Ensure accessability of https://huggingface.co/pyannote/speaker-diarization-3.1.
```bash
pip install -r requirements.txt
python separate.py
```
You can also use multi-GPUs to accelerate the whole process by modifying num_gpus in separate.py
