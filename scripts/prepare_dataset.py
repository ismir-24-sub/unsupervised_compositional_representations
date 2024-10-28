import librosa as li
import numpy as np
import yaml
import os
from tqdm.auto import tqdm
import torch
from torch import nn
import encodec
from einops import rearrange
from udls import AudioExample
from natsort import natsorted
import lmdb

from argparse import ArgumentParser


def rms_energy(audio):
    """
    Calculate the Root Mean Square (RMS) energy of an audio signal along the time axis.

    Parameters:
    - audio: numpy array of shape (..., T), where T is the duration of the audio signal.

    Returns:
    - RMS energy of the input audio.
    """
    return np.sqrt(np.mean(audio**2, axis=-1))  # RMS across the time dimension (T)


def remove_batches_with_silent_instruments(
    batch_audio: np.ndarray, rms_threshold: float = 0.001
):
    """
    Remove entire batches if any instrument in the batch has RMS energy below a given threshold, indicating silence.

    Parameters:
    - batch_audio: numpy array of shape (B, 4, T), where B is batch size, 4 is number of instruments, T is duration.
    - rms_threshold: RMS value below which the audio is considered "silent". Default is 0.01.

    Returns:
    - A new numpy array with batches removed where any instrument is silent.
    """
    # Calculate RMS energy for each instrument (axis 1) in each batch (axis 0)
    rms_values = rms_energy(batch_audio)  # Shape (B, 4)

    # Create a mask for each batch: True if all instruments have RMS energy above the threshold
    batch_silent_mask = np.any(
        rms_values < rms_threshold, axis=1
    )  # True if any instrument in the batch is silent

    # Keep only the batches where no instrument is silent
    non_silent_batches = batch_audio[~batch_silent_mask]

    # Return the filtered batch
    return non_silent_batches


class Encodec(nn.Module):

    def __init__(
        self, sr: int = 24000, bandwidth: float = 24.0, device: str = "cuda:0"
    ):
        super().__init__()
        self.sr = sr
        assert sr in [24000, 48000]
        self.device = device
        if sr == 24000:
            self.model = encodec.model.EncodecModel.encodec_model_24khz()
        else:
            self.model = encodec.model.EncodecModel.encodec_model_48khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def encode(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        x = x.to(self.device)
        codes, _ = self.model._encode_frame(x)
        codes = codes.transpose(0, 1)
        z = self.model.quantizer.decode(codes)
        return z.cpu().numpy()

    @torch.no_grad()
    def decode(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z = z.to(self.device)
        x = self.model.decoder(z)
        return x.cpu().numpy()


parser = ArgumentParser()
parser.add_argument("--topdir", "-t", type=str, required=True, default="slakh2100_flac_redux")
parser.add_argument("--subdir", "-s", type=str, required=True, default="validation")
parser.add_argument("--outdir", "-o", type=str, default=".")
parser.add_argument("--sr", type=int, default=24000)
args = parser.parse_args()

sr = args.sr
n_signal = 7 * sr
ae = Encodec(sr=sr)
topdir = args.topdir
subdir = args.subdir
jdir = os.path.join(topdir, subdir)
tracks = [f for f in os.listdir(jdir) if f.startswith("Track")]
print(len(tracks))
tjdir = [os.path.join(jdir, f) for f in tracks]

bass, drums = {}, {}
for t in tqdm(tjdir, leave=False):
    y = os.path.join(t, "metadata.yaml")
    with open(y, "r") as stream:
        data = yaml.safe_load(stream)
        stems = data["stems"]
        _drums, _bass = [], []
        for stem, stems_data in stems.items():
            inst_class = stems_data["inst_class"]
            if inst_class == "Bass":
                _bass.append(stem)
            elif inst_class == "Drums":
                _drums.append(stem)
        if len(_drums) == 0 or len(_bass) == 0:
            print("track ", t, " empty ?")
            continue
        bass[t] = _bass
        drums[t] = _drums

missing_stems = []
for t in tqdm(tjdir, leave=False):
    all_stems = bass[t] + drums[t]
    assert len(all_stems) == len(set(all_stems))
    for s in all_stems:
        if not os.path.isfile(os.path.join(t, "stems", f"{s}.flac")):
            print("missing stem ", s, " in ", t)
            missing_stems.append(t)

missing_stems = list(set(missing_stems))
print("there are ", len(missing_stems), " missing stems")

new_stems = [t for t in tjdir if t not in missing_stems]
audio = {t: {} for t in new_stems}
print(len(new_stems), "tracks left")

def load_and_mix(t, stem_list):
    tmp = 0
    for elem in stem_list[t]:
        wav, _ = li.load(os.path.join(t, "stems", f"{elem}.flac"), sr=sr)
        tmp += wav
    return tmp / len(stem_list[t]), tmp.shape

dataset_size = 200
env = lmdb.open(
    f"slakh_encodec_db_{subdir}_{n_signal}",
    map_size=int(dataset_size * (1024**3)),
)
tot_elems, filtered_elems = 0, 0
for t in tqdm(natsorted(new_stems)):
    tname = t.split("/")[-1]
    with env.begin(write=True) as txn:
        d, len_d = load_and_mix(t, drums)
        b, len_b = load_and_mix(t, bass)
        if not len_d == len_b:
            print("length of sources not the same ", t, len_d, len_b)
            continue
        factor = len_d[0] // n_signal
        d = d[: n_signal * factor].reshape((-1, 1, n_signal))
        b = b[: n_signal * factor].reshape((-1, 1, n_signal))
        all_inst = np.concatenate([d, b], axis=1)
        tot_elems += all_inst.shape[0]
        all_inst = remove_batches_with_silent_instruments(all_inst, rms_threshold=0.01)
        mix = all_inst.mean(axis=1, keepdims=True)
        all_inst = np.concatenate([all_inst, mix], axis=1)  # (b, 3, n)
        filtered_elems += all_inst.shape[0]
        batch_size, c, _ = all_inst.shape
        if batch_size > 0:
            all_inst = rearrange(all_inst, "b c n -> (b c) n")
            try:
                all_inst = ae.encode(all_inst)
            except Exception as e:
                print(e)
                continue
            all_inst = rearrange(all_inst, "(b c) d n -> b c d n", c=c)
            for idx, ai in enumerate(all_inst):
                audioexample = AudioExample()
                d, b, m = ai[0], ai[1], ai[2]
                audioexample.put("drums", d, dtype=np.float32)
                audioexample.put("bass", b, dtype=np.float32)
                audioexample.put("mix", m, dtype=np.float32)
                txn.put(f"{tname}_{idx}".encode(), bytes(audioexample))
env.close()

print(
    "total elements ",
    tot_elems,
    " filtered elements ",
    filtered_elems,
    " ratio ",
    100 * filtered_elems / tot_elems,
)
