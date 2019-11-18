Guide to setup hand-rkg conda environment

Instructions

Installing and creating conda environment:


Get the installer from here: https://docs.conda.io/en/latest/miniconda.html, install conda (make sure it is available in $path)

Redirect to github project base directory in command line

Execute:  conda env create -f conda-env.yml # this will install required project dependencies

Activate environment: conda activate hand-rkg # this will activate conda environment


Updating conda environment:


In order to update depdendencies, add that dependency in conda-env.yml file and run conda env update -f conda-env.yml



Deleting conda environment:


In order to delete conda environment:

conda deactivate

conda env remove -n hand-rkg --all
