from speechbrain.pretrained import SepformerSeparation as separator
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from speechbrain.pretrained import SpeakerRecognition

from collections import defaultdict 
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import torchaudio
import torch
import torch.nn.functional as F
import os
import json
import re
SAMPLE_RATE = 16000

CUTOFF = defaultdict(lambda: {"start": 0 * SAMPLE_RATE, "end": -0 * SAMPLE_RATE})


def separate(batch, spk2audio, turn, sv_model, spk_embs, ss_model):
    """
    Separate overlapping speech, and use speaker verification to concat segments
    """
    pad_front_leng = min(5 * SAMPLE_RATE, turn["start"])
    pad_back_leng = min(5 * SAMPLE_RATE, batch.size(-1) - turn["end"])
    separated = ss_model.separate_batch(batch[:, turn["start"] - pad_front_leng: turn["end"] + pad_back_leng]).permute(0, 2, 1)

    # This part is important, rescale the audio energy to original level to get better quality
    energy = torch.sum(torch.abs(batch[:, turn["start"] - pad_front_leng: turn["end"] + pad_back_leng]))
    hyp_energy = torch.sum(torch.abs(separated), -1) + 1e-8
    scaling_factor = energy / hyp_energy

    separated[:, 0] *= scaling_factor[0, 0]
    separated[:, 1] *= scaling_factor[0, 1]
    try:
        emb_1 = sv_model.encode_batch(separated[:, 0])[0]
        emb_2 = sv_model.encode_batch(separated[:, 1])[0]
    except RuntimeError:    # overlapping is too short
        print("Overlapping is too short, skipped")
        return
    score1 = F.cosine_similarity(emb_1, (spk_embs[turn["spk"][0]] - spk_embs[turn["spk"][1]]))
    score2 = F.cosine_similarity(emb_2, (spk_embs[turn["spk"][0]] - spk_embs[turn["spk"][1]]))

    if score1 > score2:
        spk2audio[turn["spk"][0]][:, turn["start"]: turn["end"]] = separated[:, 0, pad_front_leng: -1 * pad_back_leng or None]
        spk2audio[turn["spk"][1]][:, turn["start"]: turn["end"]] = separated[:, 1, pad_front_leng: -1 * pad_back_leng or None]
    else:
        spk2audio[turn["spk"][0]][:, turn["start"]: turn["end"]] = separated[:, 1, pad_front_leng: -1 * pad_back_leng or None]
        spk2audio[turn["spk"][1]][:, turn["start"]: turn["end"]] = separated[:, 0, pad_front_leng: -1 * pad_back_leng or None]

def find_overlap_assign(batch, diarized, pad_prev=0.5, pad_back=0.5):
    """
    WARNING, there might contain bugs when overlapping between more than two speakers.
    I do not handle this for my separation model can only handle two speakers situatiion,
    and such events is extremely rare (most of collected data are two-speaker only)
    """
    spk2audio = defaultdict(lambda: torch.zeros(batch.size()))
    spk2active = defaultdict(lambda: torch.zeros(batch.size(), dtype=torch.bool))
    overlapping = []
    cur_turn = diarized[0]
    spk2audio[cur_turn["spk"]][0, cur_turn["start"]: cur_turn["end"]] = batch[0, cur_turn["start"]: cur_turn["end"]]
    spk2active[cur_turn["spk"]][0, cur_turn["start"]: cur_turn["end"]] = True

    for turn in diarized[1:]:
        spk2audio[turn["spk"]][0, turn["start"]: turn["end"]] = batch[0, turn["start"]: turn["end"]]
        spk2active[turn["spk"]][0, turn["start"]: turn["end"]] = True
        if cur_turn["end"] > turn["start"]:
            cur_spk = cur_turn["spk"]
            cur_turn, o_end = (cur_turn, turn["end"]) if cur_turn["end"] > turn["end"] else (turn, cur_turn["end"])
            overlapping.append({"spk": (cur_spk, turn["spk"]), "start": max(turn["start"] - int(pad_prev * SAMPLE_RATE), 0), "end": min(o_end + int(pad_back * SAMPLE_RATE), batch.size(-1))})
        else:
            cur_turn = turn

    return spk2audio, spk2active, overlapping

def merge_segments(diarized, gap=1, min_dur=0.0):
    """
    Remove too short segments, and merge neighbor segments with the same speaker
    gap: max neighbor distance
    min_dur: min duration
    """
    segments = []
    prev = defaultdict(lambda: {"start": -1, "end": -1})
    spk_duration = defaultdict((lambda: 0))
    total_time = 0
    for turn, _, spk in diarized:
        spk_duration[spk] += turn.end - turn.start
        total_time += turn.end - turn.start
        if prev[spk]["start"] < 0:
            prev[spk]["start"] = turn.start
        elif turn.start - prev[spk]["end"] > gap:
            if prev[spk]["end"] - prev[spk]["start"] > min_dur:
                segments.append({"spk": spk, "start": round(prev[spk]["start"], 3), "end": round(prev[spk]["end"], 3)})
            prev[spk]["start"] = turn.start
            prev[spk]["end"] = turn.end
    
        prev[spk]["end"] = turn.end
    # check whether to add the last turn of each speaker
    for spk in prev.keys():
        if prev[spk]["end"] > 0 and prev[spk]["end"] - prev[spk]["start"] > min_dur:
            segments.append({"spk": spk, "start": round(prev[spk]["start"], 3), "end": round(prev[spk]["end"], 3)})

    # find active speaker to keep
    active_speaker = []
    for spk in spk_duration.keys():
        if spk_duration[spk] / total_time > 0.05 or spk_duration[spk] > 300:
            active_speaker.append(spk)

    # filter out non-active speaker
    filter_segments = []
    turn_end = 0
    for seg in segments:
        if seg["spk"] in active_speaker:
            seg["start"] = int(seg["start"] * SAMPLE_RATE)
            seg["end"] = int(seg["end"] * SAMPLE_RATE)
            if seg["end"] > turn_end:
                turn_end = seg["end"]
            filter_segments.append(seg)
    
    # sort turn by start time
    filter_segments.sort(key=lambda x: x["start"])
    if len(active_speaker) <= 1:
        return None, -1, len(active_speaker)
    return filter_segments, turn_end, len(active_speaker)

def get_spk_embs(spk2audio, spk2active, sv_model):
    non_overlaps = {}
    for spk in spk2audio:
        non_overlaps[spk] = spk2active[spk]
        for other_spk in spk2audio:
            if other_spk != spk:
                non_overlaps[spk] = non_overlaps[spk] & (~ spk2active[other_spk])
    spk_embs = {}
    for spk, audio in spk2audio.items():
        try:
            spk_embs[spk] = sv_model.encode_batch(torch.masked_select(audio, non_overlaps[spk])[: 10 * SAMPLE_RATE])[0]
        except RuntimeError:
            print("Too short input for sv_model when spk embs bulding")
            spk_embs[spk] = torch.zeros(1, sv_model.mods.embedding_model.fc.conv.out_channels).to(sv_model.device)

    return spk_embs

def process_data(
        wav_path, 
        sd_model, 
        ss_model, 
        sv_model, 
        out_folder,
        max_chunk_size=120,
        min_chunk_size=30,
        shift=10,
        active_ratio=0.01,
    ):
    channel, title = wav_path.split('/')[-2:]
    if os.path.exists(f"/mounted/processed/podcast/origin/{channel}"):
        return None, -1
    title = title.replace(".flac", "")
    out_path = f"{out_folder}/%s/{channel}/{title}/%04d.flac"

    # load audio
    batch, sr = torchaudio.load(wav_path)
    assert sr == SAMPLE_RATE, f"sample rate not the same: {wav_path}"

    batch = batch[:, max(CUTOFF[channel]["start"], 90): min(-90, CUTOFF[channel]["end"])]
    if batch.size(-1) < min_chunk_size * SAMPLE_RATE:
        return wav_path, -1

    # spk diarization
    diarized = sd_model({"waveform": batch, "sample_rate": SAMPLE_RATE}).itertracks(yield_label=True)
    diarized, turn_end, num_spk = merge_segments(diarized)
    torch.cuda.empty_cache()

    if diarized is None:
        return wav_path, num_spk
    else:
        batch = batch[:, : turn_end]     # truncate the tail (tuncate head is not convenience, so I just skip the head latter)
        makedirs(f"{folder}/separated/{channel}/{title}", exist_ok=True)
        makedirs(f"{folder}/origin/{channel}/{title}", exist_ok=True)

    # find overlapping and separate audio into spk2audio
    batch = batch.to(sd_model.device)
    spk2audio, spk2active, overlaps = find_overlap_assign(batch, diarized)

    spk_embs = get_spk_embs(spk2audio, spk2active, sv_model)

    for overlap in overlaps:
        separate(batch, spk2audio, overlap, sv_model, spk_embs, ss_model)
    del spk_embs

    # find active segments and save to disk
    num_segs = 0
    next_shift = max_chunk_size * SAMPLE_RATE           # normal shift size
    chunk = {"start": diarized[0]["start"], "end": diarized[0]["start"] + next_shift}
    window_shift = shift * SAMPLE_RATE                  # if not suitanble chunk meet, slide the window by window_shift
    
    spk2audio = {spk: audio.cpu() for spk, audio in spk2audio.items()}
    batch = batch.cpu()
    spk2dur = {spk: torch.sum(active[:, chunk["start"]: chunk["end"]]).item() for spk, active in spk2active.items()}

    while chunk["start"] < chunk["end"] - min_chunk_size * SAMPLE_RATE:
        active_spk = []
        for spk in spk2dur:
            if spk2dur[spk] / (chunk["end"] - chunk["start"]) > active_ratio:
                active_spk.append(spk)
        if len(active_spk) == 2:
            torchaudio.save(out_path % ("separated", num_segs), torch.stack([spk2audio[spk][0, chunk["start"]: chunk["end"]] for spk in active_spk]), sample_rate=SAMPLE_RATE)
            torchaudio.save(out_path % ("origin", num_segs), batch[:, chunk["start"]: chunk["end"]], sample_rate=SAMPLE_RATE)
            num_segs += 1
            
            chunk["start"] += next_shift
            chunk["end"] = min(chunk["end"] + next_shift, batch.size(-1))
            for spk in spk2dur:
                spk2dur[spk] = torch.sum(spk2active[spk][:, chunk["start"]: chunk["end"]]).item()
        else:
            for spk in spk2dur:
                spk2dur[spk] += torch.sum(spk2active[spk][:, chunk["end"]: chunk["end"] + window_shift]) - torch.sum(spk2active[spk][:, chunk["start"]: chunk["start"] + window_shift])
            chunk["start"] += window_shift
            chunk["end"] = min(chunk["end"] + window_shift, batch.size(-1))
    del batch
    del spk2audio
    torch.cuda.empty_cache()
    return wav_path, num_spk

if __name__ == "__main__":
    audio_path = '/content/audio'
    out_folder = "/content/test/"
    num_gpus = 1

    os.makedirs(out_folder + "/origin/", exist_ok=True)
    os.makedirs(out_folder + "/separated/", exist_ok=True)

    # speaker diarization model
    sd_name = "pyannote/speaker-diarization-3.1"
    sd_model = [Pipeline.from_pretrained(sd_name).to(torch.device(f"cuda:{i}")) for i in range(num_gpus)]

    # source separation model
    ss_ckpt = "./ss_model/"
    ss_model = [separator.from_hparams(source=ss_ckpt, hparams_file="hyperparams.yaml", run_opts={"device": torch.device(f"cuda:{i}")}).eval() for i in range(num_gpus)]

    # speakder verification model
    sv_name = "speechbrain/spkrec-ecapa-voxceleb"
    sv_model = [SpeakerRecognition.from_hparams(source=sv_name, savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device": torch.device(f"cuda:{i}")}) for i in range(num_gpus)]

    with torch.no_grad():
        with open(f"{out_folder}/infor.txt", "w") as f:
            for path, num_spk in Parallel(n_jobs=-1, backend="threading", return_as="generator")(delayed(process_data)(wav_path, sd_model[i % num_gpus], ss_model[i % num_gpus], sv_model[i % num_gpus], out_folder) for i, wav_path in enumerate(tqdm(glob(audio_path + "/*.flac", recursive=True)))):
                if path:
                    f.write(f"{path}\t{num_spk}\n")
