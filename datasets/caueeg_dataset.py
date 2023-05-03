import os
import json
from copy import deepcopy
import pyedflib

import numpy as np
import pyarrow.feather as feather
import torch
from torch.utils.data import Dataset


class CauEegDataset(Dataset):
    """PyTorch Dataset Class for CAUEEG Dataset.

    Args:
        root_dir (str): Root path to the EDF data files.
        data_list (list of dict): List of dictionary for the data.
        load_event (bool): Determines whether to load event information or not for saving loading time.
        file_format (str): Determines which file format is used among of EDF, PyArrow Feather, and NumPy memmap.
        transform (callable): Optional transform to be applied on each data.
    """

    def __init__(self, root_dir: str, data_list: list, load_event: bool, file_format: str = "edf", transform=None):
        if file_format not in ["edf", "feather", "memmap", "np"]:
            raise ValueError(
                f"{self.__class__.__name__}.__init__(file_format) "
                f"must be set to one of 'edf', 'feather', 'memmap' and 'np'"
            )

        self.root_dir = root_dir
        self.data_list = data_list
        self.load_event = load_event
        self.file_format = file_format
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # annotation
        sample = deepcopy(self.data_list[idx])

        # signal
        sample["signal"] = self._read_signal(sample)

        # event
        if self.load_event:
            sample["event"] = self._read_event(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _read_signal(self, anno):
        if self.file_format == "edf":
            return self._read_edf(anno)
        elif self.file_format == "feather":
            return self._read_feather(anno)
        else:
            return self._read_memmap(anno)

    def _read_edf(self, anno):
        edf_file = os.path.join(self.root_dir, f"signal/edf/{anno['serial']}.edf")
        signal, signal_headers, _ = pyedflib.highlevel.read_edf(edf_file)
        return signal

    def _read_feather(self, anno):
        fname = os.path.join(self.root_dir, f"signal/feather/{anno['serial']}.feather")
        df = feather.read_feather(fname)
        return df.values.T

    def _read_memmap(self, anno):
        fname = os.path.join(self.root_dir, f"signal/memmap/{anno['serial']}.dat")
        signal = np.memmap(fname, dtype="int32", mode="r").reshape(21, -1)
        return signal

    def _read_np(self, anno):
        fname = os.path.join(self.root_dir, f"signal/{anno['serial']}.npy")
        return np.load(fname)

    def _read_event(self, m):
        fname = os.path.join(self.root_dir, "event", m["serial"] + ".json")
        with open(fname, "r") as json_file:
            event = json.load(json_file)
        return event
