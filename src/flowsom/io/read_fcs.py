from __future__ import annotations

import csv
import os
import re

import anndata as ad
import flowio
import numpy as np
import pandas as pd
import readfcs


def read_FCS(filepath):
    """Reads in an FCS file.

    :param filepath: An array containing a full path to the FCS file
    :type filepath: str
    """
    try:
        f = readfcs.read(filepath, reindex=True)
        f.var.n = f.var.n.astype(int)
        f.var = f.var.sort_values(by="n")
        f.uns["meta"]["channels"].index = f.uns["meta"]["channels"].index.astype(int)
        f.uns["meta"]["channels"] = f.uns["meta"]["channels"].sort_index()
    except ValueError:
        f = readfcs.read(filepath, reindex=False)
        markers = {
            str(re.sub("S$", "", re.sub("^P", "", string))): f.uns["meta"][string]
            for string in f.uns["meta"].keys()
            if re.match("^P[0-9]+S$", string)
        }
        fluo_channels = list(markers.keys())
        non_fluo_channels = {
            i: f.uns["meta"]["channels"]["$PnN"][i] for i in f.uns["meta"]["channels"].index if i not in fluo_channels
        }
        index_markers = dict(markers, **non_fluo_channels)
        f.var.rename(index=index_markers, inplace=True)
        f.uns["meta"]["channels"]["$PnS"] = [index_markers[key] for key in f.uns["meta"]["channels"].index]
    return f


def read_csv(filepath, spillover=None, **kwargs):
    """Reads in a CSV file."""
    ff = ad.read_csv(filepath, **kwargs)
    ff.var = pd.DataFrame(
        {"n": range(ff.shape[1]), "channel": ff.var_names, "marker": ff.var_names}, index=ff.var.index
    )
    if spillover is not None:
        ff.uns["meta"]["SPILL"] = pd.read_csv(spillover)
    return ff


def read_FCS_numpy(filepath):
    """Reads in an FCS file.

    :param filepath: An array containing a full path to the FCS file
    :type filepath: str
    """
    fcs_data = flowio.FlowData(filepath)
    return np.reshape(fcs_data.events, (-1, fcs_data.channel_count))


def read_csv_dataset(file_path):
    """Reads in a CSV file."""
    array_list = []
    dir_path = os.path.dirname(file_path)

    with open(file_path, newline="") as f_in:
        reader = csv.reader(f_in)
        for row in reader:
            try:
                array_list.append(read_FCS_numpy(f"{dir_path}/{row[1]}"))
            except FileNotFoundError:
                pass

    return np.concatenate(array_list) if array_list else None
