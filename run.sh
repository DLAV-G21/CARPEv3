#!/bin/bash
#SBATCH --chdir /home/plumey/ProjectRepository
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023
#SBATCH --time 12:00:00

source ../venvs/venv-g21/bin/activate
python3 train.py baseline --coco_path carpe_data --batch_size 32 --num_queries 25 --set_cost_class 10