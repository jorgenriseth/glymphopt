rule simulate_data:
    output:
        "data/sim.hdf"
    shell:
        "python scripts/simulate_data.py"
        " --output {output}"
        " --resolution 512"

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
        resolution = 256
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
        resolution = 100
    shell:
        "python scripts/mri2fem.py"
        " --inputfiles {input.data}"
        " --timestamps {input.timestamps}"
        " --output {output}"
        " --resolution {params.resolution}"

        
