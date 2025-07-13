with open("subjects.txt", "r") as f:
    SUBJECTS = [line.strip() for line in f.readlines()]

def get_optimal(file, key):
  import pandas as pd
  dframe = pd.read_csv(file, sep=";")
  return float(dframe.loc[dframe["funceval"].idxmin()][key])

rule all:
    input:
        #[f"results/{subject}.json" for subject in SUBJECTS if exists(f"mri_processed_data/{subject}/modeling/resolution32/data.hdf") ]
        [f"results/{subject}_singlecomp_gridsearch.csv" for subject in SUBJECTS if exists(f"mri_processed_data/{subject}/modeling/resolution32/data.hdf") ],
        [f"results/{subject}_twocomp_gridsearch.csv" for subject in SUBJECTS if exists(f"mri_processed_data/{subject}/modeling/resolution32/data.hdf") ]


rule optimize_singlecompartment:
    input:
        "mri_processed_data/{subject}/modeling/resolution32/data.hdf"
    output:
        "results/{subject}_singlecomp_minimization.json"
    shell:
        "python scripts/diffusion_reaction_minimization.py"
        " -i {input}"
        " -o {output}"

rule gridsearch_singlecompartment:
    input:
        "mri_processed_data/{subject}/modeling/resolution32/data.hdf"
    output:
        "results/{subject}_singlecomp_gridsearch.csv"
    shell:
        "python scripts/diffusion_reaction_gridsearch.py"
        " -i {input}"
        " -o {output}"

rule gridsearch_twocompartment:
    input:
        "mri_processed_data/{subject}/modeling/resolution32/data.hdf"
    output:
        "results/{subject}_twocomp_gridsearch.csv"
    resources:
        runtime="30h"
    shell:
        "python scripts/multicomp_gridsearch.py"
        " -i {input}"
        " -o {output}"


rule run_optimal_singlecompartment:
  input:
    "results/{subject}_singlecomp_gridsearch.csv",
    data="mri_processed_data/{subject}/modeling/resolution32/data.hdf",
  output:
    "results/{subject}_singlecomp_optimal.hdf",
  params:
    alpha = lambda wc: get_optimal(f"results/{wc.subject}_singlecomp_gridsearch.csv", "a"),
    r = lambda wc: get_optimal(f"results/{wc.subject}_singlecomp_gridsearch.csv", "r")
  shell:
    "python scripts/singlecompartment.py"
    " -i {input.data}"
    " -o {output}"
    " -a {params.alpha}"
    " -r {params.r}"

rule run_optimal_multicompartment:
  input:
    "results/{subject}_multicomp_gridsearch.csv",
    "mri_processed_data/{subject}/modeling/resolution32/data.hdf",
  output:
    "results/{subject}_multicomp_optimal.hdf",
    "results/{subject}_multicomp_optimal.xdmf",
  params:
    gamma = lambda wc: get_optimal(f"results/{wc.subject}_twocomp_gridsearch.csv", "gamma"),
    t_pb = lambda wc: get_optimal(f"results/{wc.subject}_twocomp_gridsearch.csv", "t_pb")
  shell:
    "python scripts/multicomp_run.py"
    " -i {input}"
    " --gamma {params.gamma}"
    " --t_pb {params.t_pb}"

rule errortable:
  input:
    data="mri_processed_data/{subject}/modeling/resolution32/data.hdf",
    single="results/{subject}_singlecomp_optimal.hdf",
    multi="results/{subject}_multicomp_optimal.hdf",
  output:
    "results/{subject}_errortable.csv"
  shell:
    "python scripts/create_errortable.py"
    " -d {input.data}"
    " --single {input.single}"
    " --multi {input.multi}"
