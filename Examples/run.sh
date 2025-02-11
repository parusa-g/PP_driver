#!/bin/bash
#SBATCH --job-name=MoTe2
#SBATCH --cluster=merlin6
#SBATCH --partition=hourly
#SBATCH --time=00:10:00
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --hint=nomultithread
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=1
#SBATCH --nodes=1
#==================================================

PW=/psi/home/parusa_g/lmi-qe-mod/bin/pw.x

[ ! -d "outfiles" ] && mkdir outfiles
[ ! -d "outdir" ] && mkdir outdir
[ ! -d "slurm" ] && mkdir slurm

srun $PW -nk 16 -nd 16 < inp/scf.in > outfiles/scf.out
