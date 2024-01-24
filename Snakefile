RESOLUTION_SIM = 1024
RESOLUTION_MRI = 64
RESOLUTION_MESH = 31

rule simulate_data:
    output:
        "data/sim.hdf"
    params:
        resolution = RESOLUTION_SIM
    shell:
        "python scripts/simulate_data.py"
        " --output {output}"
        " --resolution {params.resolution}"

NUM_SAMPLES = 5
MEASURE_TIMES = [idx / NUM_SAMPLES for idx in range(NUM_SAMPLES+1)]

rule subsample_data:
    input:
        "data/sim.hdf"
    output:
        "data/sim_sampled.hdf"
    shell:
        "python scripts/generate_data_1D.py"
        " --input {input}"
        " --output {output}"
        " --sampletimes {MEASURE_TIMES}"
        " --fem_family 'CG'"
        " --fem_degree 1"

CONCENTRATIONS = expand(
    "data/concentration_{idx}.npz",
    idx=range(NUM_SAMPLES+1)
)

rule measure_data:
    input:
        "data/sim_sampled.hdf"
    output:
        concentrations = CONCENTRATIONS,
        timestamps = "data/timestamps.txt"
    params:
        resolution = RESOLUTION_MRI
    shell:
        "python scripts/measure.py"
        " --infile {input}"
        " --resolution {params.resolution}"
        " --timestamps_out {output.timestamps}"


rule mri2fenics:
    input:
        data = CONCENTRATIONS,
        timestamps = "data/timestamps.txt"
    output:
        "data/data.hdf"
    params: 
        resolution = RESOLUTION_MESH
    shell:
        "python scripts/mri2fem.py"
        " --inputfiles {input.data}"
        " --timestamps {input.timestamps}"
        " --output {output}"
        " --resolution {params.resolution}"


rule inverse_diffusion_coefficient:
    input:
        "data/data.hdf",
    output:
        "results/optimal.hdf"
    shell:
        "python scripts/optimal_diffusion_coefficient.py"
        " --input {input}"
