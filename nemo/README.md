Image: nvidia/nemo:24.05 (from NGC)
train:
Modify the path to pretrained model and dataset path.
```bash
bash SFT.sh
```
inference:
Modify the path to the model checkpoint in launch.sh, and run
```bash
bash launch.sh
```
Open another terminal
```bash
python interactive.py
```
Now you can test the model 