#!/bin/bash
#SBATCH --job-name hello_world_test
#SBATCH --cpus-per-task 4
#SBATCH --output ./outputs/%j.output.out
#SBATCH --error ./outputs/errors/%j.error.out
#SBATCH --chdir /projects/nlpcod/care_nlp_psych
#SBATCH -p debug
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tphaterp@student.ubc.ca
#SBATCH --nodelist=hpc[1-3]
#####run commands to execute
echo  'We are starting main.sh'
source ./basic_env/bin/activate

if [[ $? -eq 0 ]]; then
    echo 'Virtual environment activated successfully'
else
    echo 'Error activating virtual environment'
    exit 1
fi

python3 -V
python3 ./hello_world.py
echo 'We ran ./hello_world.py succesfully'
#python3 ./pytorch_script_gpu_avail.py
deactivate
echo 'We finished main.sh'