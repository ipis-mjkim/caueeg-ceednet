# Load some packages
import os
from pathlib import Path
import glob
import pyedflib
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm.auto import tqdm

from pipeline import trim_trailing_zeros

if __name__ == "__main__":
    signal_root = os.path.join(Path(__file__).parents[1].absolute(), "local/dataset/caueeg-dataset/signal")

    os.makedirs(os.path.join(signal_root, "feather"), exist_ok=True)

    for f in tqdm(glob.glob(os.path.join(signal_root, "edf/*.edf"))):
        # file name
        serial = f.split(".edf")[0][-5:]

        # load signal
        signals, signal_headers, edf_header = pyedflib.highlevel.read_edf(f)
        signals = trim_trailing_zeros(signals)
        signals = signals.astype("int32")

        df = pd.DataFrame(data=signals.T, columns=[s_h["label"] for s_h in signal_headers], dtype=np.int32)
        feather.write_feather(df, os.path.join(signal_root, "feather", serial + ".feather"))
