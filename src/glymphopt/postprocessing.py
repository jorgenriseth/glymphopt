import dolfin as df
import numpy as np
import pandas as pd


def compute_mri_segment_stats(concentration, seg, label):
    voxelcount = (seg == label).sum()
    tracer_amount = np.nansum(concentration[seg == label]) * 0.5**3
    mean_concentration = np.nanmean(concentration[seg == label])

    return {
        "label": label,
        "voxelcount": voxelcount,
        "volume": voxelcount * 0.5**3,
        "tracer_amount": tracer_amount,
        "mean_concentration": mean_concentration,
    }


def compute_mesh_segment_stats(u, parcellations, dx, label):
    cellcount = (parcellations.array() == label).sum()
    volume = df.assemble(1 * dx(int(label)))
    tracer_amount = df.assemble(u * dx(int(label)))
    return {
        "label": label,
        "cellcount": cellcount,
        "volume": volume,
        "tracer_amount": tracer_amount,
        "mean_concentration": tracer_amount / volume,
    }


def compute_mesh_regionwise_statistics(u, parcellations):
    parcellation_labels = np.unique(parcellations.array())
    domain = u.function_space().mesh()
    dx = df.Measure("dx", domain=domain, subdomain_data=parcellations)
    records = [
        compute_mesh_segment_stats(u, parcellations, dx, label)
        for label in map(int, parcellation_labels)
    ]
    return pd.DataFrame.from_records(records)


def table_to_dict_map(table, keycolumn: str, valuecolumn: str):
    keys = table[keycolumn]
    assert len(keys) == len(np.unique(keys)), (
        f"keycolumn '{keycolumn}' contains duplicate entries"
    )
    return {label: value for label, value in zip(table[keycolumn], table[valuecolumn])}


def map_stats_to_segmentation(
    segmentation: np.ndarray,
    table: pd.DataFrame,
    keycolumn: str,
    valuecolumn: str,
):
    data = table_to_dict_map(table, keycolumn, valuecolumn)
    labels = data.keys()
    segment_data = np.nan * np.zeros(segmentation.shape)
    for label in labels:
        segment_data[segmentation == label] = data[int(label)]
    return segment_data
