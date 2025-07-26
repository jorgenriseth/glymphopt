from pathlib import Path


import click
import dolfin as df
import pantarei as pr
import pandas as pd

from glymphopt.io import read_function_data
from glymphopt.postprocessing import compute_mesh_regionwise_statistics
from glymphopt.utils import prepend_info


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


if __name__ == "__main__":
    main()
