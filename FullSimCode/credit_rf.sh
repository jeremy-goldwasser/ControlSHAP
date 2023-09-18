#!/bin/bash
#$ -N jobname
#$ -j y    # join output and error
module load python/3.11.5  # <- load a recent version of Python 3
source venv_shap/bin/activate
python -u sim_credit_rf.py




