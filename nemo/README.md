Image: nvidia/nemo:24.05 (from NGC)
train:
Modify the path to pretrained model and dataset path in SFT.sh.
```bash
bash SFT.sh
```
inference:
Modify the path to the model checkpoint in launch.sh, and run launch.sh to launch inference server.
```bash
bash launch.sh
```
Open another terminal
```bash
python interactive.py
```
Now you can push API call to the inference server in interactive mode.
