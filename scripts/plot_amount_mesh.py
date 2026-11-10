from pathlib import Path

import click
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_amount_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = pd.read_csv(path, usecols=["time", "amount_mesh"])
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise click.ClickException(
            f"Cannot read {path}: expected CSV columns 'time' and 'amount_mesh'. {error}"
        ) from error
    if data.empty:
        raise click.ClickException(f"{path}: CSV contains no recorded states.")
    try:
        values = data[["time", "amount_mesh"]].to_numpy(dtype=float)
    except ValueError as error:
        raise click.ClickException(
            f"{path}: 'time' and 'amount_mesh' must contain numeric values."
        ) from error
    if not np.isfinite(values).all():
        raise click.ClickException(
            f"{path}: 'time' and 'amount_mesh' must contain only finite values."
        )
    time, amount = values.T
    if np.any(np.diff(time) <= 0):
        raise click.ClickException(f"{path}: 'time' must be strictly increasing.")
    return time / 3600, amount


@click.command()
@click.option(
    "--input", "-i", "input_paths",
    type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
    multiple=True, help="Simulation CSV; repeat for multiple curves.",
)
@click.option("--label", multiple=True, help="Curve label; repeat once per input (defaults to filename stems).")
@click.option("--data", "data_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Optional measured-data CSV, plotted as crosses.")
@click.option(
    "--output", "-o", "output_path",
    type=click.Path(dir_okay=False, path_type=Path), required=True,
)
def main(input_paths: tuple[Path, ...], label: tuple[str, ...], data_path: Path | None, output_path: Path):
    """Plot mesh amounts (mmol) from simulation CSVs against time (hours)."""
    if output_path.suffix.lower() != ".png":
        raise click.BadParameter("Use a .png output filename.", param_hint="--output")
    if label and len(label) != len(input_paths):
        raise click.BadParameter("Provide exactly one --label per --input.", param_hint="--label")
    sources = list(input_paths) + ([data_path] if data_path is not None else [])
    if any(path.resolve() == output_path.resolve() for path in sources):
        raise click.BadParameter(
            "Output must differ from the input CSV.", param_hint="--output"
        )
    curves = [read_amount_mesh(path) for path in input_paths]
    measured = read_amount_mesh(data_path) if data_path is not None else None
    labels = label or tuple(path.stem for path in input_paths)
    fig, ax = plt.subplots()
    try:
        for (time_hours, amount), curve_label in zip(curves, labels):
            ax.plot(time_hours, amount, linewidth=1.5, label=curve_label)
        if measured is not None:
            ax.plot(*measured, "kx", label="Data")
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel("Total mesh solute amount [mmol]")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, format="png", dpi=150)
    except (OSError, ValueError) as error:
        raise click.ClickException(f"Cannot save {output_path}: {error}") from error
    finally:
        plt.close(fig)
    click.echo(f"Saved {output_path}")


if __name__ == "__main__":
    main()
