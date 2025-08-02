
## Setup
```bash
mkdir -p deps/
mkdir git clone -b boundary-cell-refinement \
  git@github.com:jorgenriseth/gMRI2FEM.git deps/gmri2fem
pixi install
```

To copy necessary input files from another source, without interfering with everyone elses work:
- Start by creating a `subjects.txt`-file listing the subjects of the study.
- Copy all necessary files from the source directory:
```bash
export SOURCEDIR=[full path to directory where mri_dataset and mri_processed_data is located]
./copy-deps.sh
```

If on a cluster with slurm execution, you need to create output directories first.
```bash
mkdir jobs logs
```
Pull the singularity container:
```bash
singularity build glymphopt.sif jorgenriseth/glymphopt
```
or build locally with the help of docker,
```bash
docker build -t glymphopt .; 
apptainer build glymphopt.sif docker-daemon:glymphopt:latest
```

