from pathlib import Path


import click
from gmri2fem.utils import prepend_info
import numpy as np
import dolfin as df
import pantarei as pr
import pandas as pd

from glymphopt.io import read_function_data


@click.command()
@click.option("--datapath", type=Path, required=True)
@click.option("--singlecomp", type=Path, required=True)
@click.option("--twocomp", type=Path, required=True)
@click.option("--output", "-o", type=Path, required=True)
def main(datapath, singlecomp, twocomp, output):
    with df.HDF5File(df.MPI.comm_world, str(datapath), "r") as hdf:
        domain = pr.read_domain(hdf)
        parcellations = df.MeshFunction("size_t", domain, domain.topology().dim())
        hdf.read(parcellations, "parcellations")

    td, Yd = read_function_data(datapath, domain, "concentration")
    _, Ym_singlecomp = read_function_data(singlecomp, domain, "concentration")
    _, Ym_twocomp = read_function_data(twocomp, domain, "concentration")

    session_dataframes = []
    for idx, (y_data, y_single, y_two) in enumerate(zip(Yd, Ym_singlecomp, Ym_twocomp)):
        dframe_data = compute_mesh_regionwise_statistics(y_data, parcellations)
        dframe_singlecomp = compute_mesh_regionwise_statistics(y_single, parcellations)
        dframe_twocomp = compute_mesh_regionwise_statistics(y_two, parcellations)

        dframe_singlecomp["error"] = (
            dframe_singlecomp["mean_concentration"] - dframe_data["mean_concentration"]
        )
        dframe_twocomp["error"] = (
            dframe_twocomp["mean_concentration"] - dframe_data["mean_concentration"]
        )
        dframe_combined = dframe_data.merge(
            dframe_singlecomp,
            on=["label", "cellcount", "volume"],
            suffixes=("_data", ""),
        ).merge(
            dframe_twocomp,
            on=["label", "cellcount", "volume"],
            suffixes=("_singlecomp", "_twocomp"),
        )
        session_dataframes.append(
            prepend_info(dframe_combined, session=f"ses-{idx + 1:02d}")
        )
    dframe = pd.concat(session_dataframes, ignore_index=True)
    dframe.to_csv(
        output,
        sep=";",
        index=False,
    )


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


if __name__ == "__main__":
    main()
