import datetime

import numpy as np
from torch.utils.data import default_collate
import torch
from lmdb import Environment  # type: ignore
from udls import AudioExample  # type: ignore
import gin


TensorType = torch.Tensor | np.ndarray


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lmdb_path: str,
        readonly: bool = True,
        data_key: str = "mix",
    ):
        self._path = lmdb_path
        self._env = Environment(lmdb_path, readonly=readonly, create=False, lock=False)
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        self._data_key = data_key
        self.print_duration()
        return

    @property
    def data_key(self) -> str:
        return self._data_key

    @property
    def keys(self) -> list[bytes]:
        return self._keys

    @property
    def path(self) -> str:
        return self._path

    @property
    def env(self) -> Environment:
        return self._env
    
    def print_duration(self):
        pass

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, TensorType | str | int]:
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            ae = AudioExample(txn.get(key))
            return dict(data=ae.get(self.data_key))


class MultiSourceDataset(BaseDataset):
    def __init__(
        self,
        lmdb_path: str,
        readonly: bool = True,
        sources: list[str] = ["drums", "bass", "piano"],
    ):
        self._sources = sources
        super().__init__(lmdb_path, readonly)   
        return

    @property
    def sources(self) -> list[str]:
        return self._sources
    
    def print_duration(self):
        pass

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, TensorType | str | int]:
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            ae = AudioExample(txn.get(key)).as_dict()
            if self.data_key in ae.keys():
                sources = {"data": ae[self.data_key]}
            else:
                sources = {}
            for source in self.sources:
                sources[source] = ae[source]
            return sources


def encodec_duration(data, n_elems, tag=""):
    *_, time = data["data"].shape
    n_seconds = (time / 75) * n_elems
    total_dur = datetime.timedelta(seconds=n_seconds)
    print(f"Total duration {tag}: {total_dur}")


@gin.configurable
class SLK_TRAIN_ENCODEC_DBP(MultiSourceDataset):
    def __init__(self, lmdb_path: str = "/data/giovanni/datasets/slakh/proper_slakh_encodec_dbp_train_168000", sources: list[str] = ["drums", "bass", "piano"]):
        super().__init__(lmdb_path, sources=sources)

    def print_duration(self):
        data = self.__getitem__(index=0)
        encodec_duration(data, len(self), tag="SLK_TRAIN_ENCODEC_DBP")
        
@gin.configurable
class SLK_VALID_ENCODEC_DBP(MultiSourceDataset):
    def __init__(self, lmdb_path: str = "/data/giovanni/datasets/slakh/proper_slakh_encodec_dbp_validation_168000", sources: list[str] = ["drums", "bass", "piano"]):
        super().__init__(lmdb_path, sources=sources)

    def print_duration(self):
        data = self.__getitem__(index=0)
        encodec_duration(data, len(self), tag="SLK_VAL_ENCODEC_DBP")

@gin.configurable
class SLK_TRAIN_ENCODEC_DB(MultiSourceDataset):
    def __init__(self, lmdb_path: str = "/data/giovanni/datasets/slakh/proper_slakh_encodec_db_train_168000", sources: list[str] = ["drums", "bass"]):
        super().__init__(lmdb_path, sources=sources)
    
    def print_duration(self):
        data = self.__getitem__(index=0)
        encodec_duration(data, len(self), tag="SLK_TRAIN_ENCODEC_DB")

@gin.configurable
class SLK_VALID_ENCODEC_DB(MultiSourceDataset):
    def __init__(self, lmdb_path: str = "/data/giovanni/datasets/slakh/proper_slakh_encodec_db_validation_168000", sources: list[str] = ["drums", "bass"]):
        super().__init__(lmdb_path, sources=sources)

    def print_duration(self):
        data = self.__getitem__(index=0)
        encodec_duration(data, len(self), tag="SLK_VAL_ENCODEC_DB")


class SLK_TRAIN_M2L_DBP(MultiSourceDataset):
    pass


class SLK_VALID_M2L_DBP(MultiSourceDataset):
    pass


@gin.configurable
def generator(seed: int = 23):
    return torch.Generator().manual_seed(seed)


@gin.configurable
def collate_crop(batch: list[dict[str, torch.Tensor]], target_len: int = 512, max_len: int = 525) -> dict[str, torch.Tensor]:
    data = default_collate(batch)
    rnd_idx = torch.randint(0, max_len - target_len, (1,)).item()
    return {k: v[..., rnd_idx : rnd_idx + target_len] for k, v in data.items()}

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from soundfile import write
    from models import Encodec

    slk_train_enc_dbp = SLK_TRAIN_ENCODEC_DB()

    enc = Encodec(device="cpu")

    dl = DataLoader(slk_train_enc_dbp, batch_size=1, collate_fn=collate_crop)
    for i, batch in enumerate(dl):
        print(batch["data"].shape)
        audio = enc.decode(batch["data"])
        print(audio.shape)
        # write(f"test_{i}.wav", audio[0].squeeze(), enc.sr)
        break