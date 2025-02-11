import os 
import numpy as np
import f90nml
import subprocess
#===============================================================================
class ExtractPotential(object):
   def __init__(self,mpicmd,exe_path,prefix,outdir,plot_num,iflag,latvec_file,atoms_file,inp_prefix,out_prefix,pot_prefix):
      '''mpicmd: mpi command
         exe_path: path to the executable pp.x
         prefix: prefix from pw.x calculations
         outdir: directory where pw.x outputs are stored
         plot_num: 1 = total potential 
                   2 = local ionic potential 
                  11 = local ionic + Hartree potential
         iflag: 0 = with spherical average
                1 = without spherical average
         latvec_file: file where each row contains the 3D lattice vector (alat unit)
         atoms_file: file containing atomic labels and positions in crystal coordinate
         inp_prefix: prefix for the input files
         out_prefix: prefix for the output files
         pot_prefix: prefix for the potential files
         '''
      
      self.mpicmd = mpicmd
      self.exe = exe_path + '/pp.x'
      self.prefix = prefix
      self.outdir = outdir
      self.plot_num = plot_num
      self.iflag = iflag
      self.file_potential = prefix
      
      latvec = np.loadtxt(latvec_file)
      atompos_frac = self.ReadAtomsFile(atoms_file)
      self.atompos_cart = atompos_frac.dot(latvec)

      if self.plot_num == 1:
         self.file_potential += '_totalPotential'
      elif self.plot_num == 2:
         self.file_potential += '_localIonicPotential'
      elif self.plot_num == 11:
         self.file_potential += '_localIonicPlusHartreePotential'

      self.inp_prefix = inp_prefix
      self.out_prefix = out_prefix
      self.pot_prefix = pot_prefix
#-------------------------------------------------------------------------------
   def ReadAtomsFile(self,atoms_file):
      '''atoms_file: file containing atomic labels and positions in crystal coordinate
                     e.g.: Si 0.00000 0.00000 0.00000
      '''

      with open(atoms_file,'r') as f:
         lines = f.readlines()

      self.atom_labels = []
      atom_positions = []
      for line in lines:
         data = line.split()
         self.atom_labels.append(data[0])
         atom_positions.append([float(data[1]),float(data[2]),float(data[3])])

      return np.array(atom_positions)
#-------------------------------------------------------------------------------
   def WriteInputPPfile(self,file_inp,atom_label='',atom_pos=[],vec=[],nx=1000,initialized=True):
      '''atom_label: atomic label
         atom_pos: 3D atomic position (alat unit), determines the origin of the 1D plot
         vec: 3D-vector that determines the direction of the 1D plot (alat unit)'''

      filplot = self.pot_prefix + self.file_potential + '.dat'
      
      if not initialized:
         inputpp = {'prefix': self.prefix,
                    'outdir': self.outdir,
                    'plot_num': self.plot_num,
                    'filplot': filplot}

         inp = {'INPUTPP': inputpp}

      else:
         output_format = 0

         if self.iflag == 0:
            suffix = '_1Dplot_withSphericalAverage.dat'
         elif self.iflag == 1:
            suffix = '_1Dplot_withoutSphericalAverage.dat'

         fileout = self.pot_prefix + self.file_potential + '_' + atom_label + suffix

         plot = {'filepp(1)': filplot,
                 'iflag': iflag,
                 'output_format': output_format,
                 'x0(1)': atom_pos[0],
                 'x0(2)': atom_pos[1],
                 'x0(3)': atom_pos[2],
                 'e1(1)': vec[0],
                 'e1(2)': vec[1],
                 'e1(3)': vec[2],
                 'nx': nx,
                 'fileout': fileout
                }

         inp = {'INPUTPP': {},
                'PLOT': plot}

      with open(file_inp, 'w') as nml_file:
         f90nml.write(inp, nml_file)
#-------------------------------------------------------------------------------
   def FindNearestNeighborVector(self,atomref_id):
      '''atomref_id: index of the reference atom'''

      # Find the center atom
      r0 = self.atompos_cart[atomref_id,:]

      # Find the nearest neighbor
      dist = 1e10
      for i in range(self.atompos_cart.shape[0]):
         if i != atomref_id:
            r1 = self.atompos_cart[i,:]
            d = np.linalg.norm(r1 - r0)
            if d < dist:
               dist = d
               nearest_neighbor = i

      nnvec = self.atompos_cart[nearest_neighbor,:] - r0

      return nearest_neighbor, nnvec
#-------------------------------------------------------------------------------   
   def WriteRunScript(self,file_in,file_out,atomref_id='',initialized=True):
      
      if not initialized:
         run_script = f'run_init.sh'
      else:
         run_script = f'run_atom#{atomref_id}.sh'

      with open(run_script,'w') as f:
         f.write('#!/bin/bash\n')
         f.write(f'#SBATCH --job-name=atom_{atomref_id}\n')
         f.write('#SBATCH --cluster=merlin6\n')
         f.write('#SBATCH --partition=hourly\n')
         f.write('#SBATCH --time=00:05:00\n')
         f.write('#SBATCH --output=slurm/%j.out\n')
         f.write('#SBATCH --error=slurm/%j.err\n')
         f.write('#SBATCH --hint=nomultithread\n')
         f.write('#SBATCH --ntasks=6\n')
         f.write('#SBATCH --ntasks-per-core=1\n')
         f.write('#SBATCH --nodes=1\n')
         f.write('\n')
         f.write(f'{self.mpicmd} {self.exe} < {file_in} > {file_out}\n')

#-------------------------------------------------------------------------------
   def WriteInputs(self,atomref_ids):
      file_inp = self.inp_prefix + prefix + '.pp.inp'
      file_out = self.out_prefix + prefix + '.pp.out'

      self.WriteInputPPfile(file_inp, initialized=False)
      self.WriteRunScript(file_inp,file_out,initialized=False)
      
      for i in range(len(atomref_ids)):
         atomref_id = atomref_ids[i]
         
         atom_label = 'Atom#' + str(atomref_id)
         atom_label += '_' + self.atom_labels[atomref_id]

         nn_id, nnvec = self.FindNearestNeighborVector(atomref_id)

         print(f'Prepare an input for atom # {atomref_id}: {self.atom_labels[atomref_id]}')
         print(f'Nearest neighbor atom: {self.atom_labels[nn_id]}, # {nn_id}')
         print(f'Nearest neighbor vector: {nnvec[0]:.6f}, {nnvec[1]:.6f}, {nnvec[2]:.6f}')

         file_inp = self.inp_prefix + self.prefix + '_' + atom_label + '.pp.inp'
         file_out = self.out_prefix + self.prefix + '_' + atom_label + '.pp.out'

         vec = 0.5 * nnvec

         self.WriteInputPPfile(file_inp,atom_label,self.atompos_cart[atomref_id,:],vec)
         self.WriteRunScript(file_inp,file_out,atomref_id)
#-------------------------------------------------------------------------------
   def run(self,atomref_ids):
      self.WriteInputs(atomref_ids)
      
      init_is_done = False
      
      init_out = self.out_prefix + self.prefix + '.pp.out'
      if not os.path.exists(init_out):
         print('Running the initialization calculation')
         os.system('sbatch run_init.sh') 
      else:
         print('Initialization calculation is already done')
      
      while not init_is_done:
         try:
            if os.path.exists(init_out):
               output = subprocess.check_output(f'grep "JOB DONE" {init_out}', shell=True)
               init_is_done = True
         except:
            init_is_done = False
      
      for i in range(len(atomref_ids)):
         os.system(f'sbatch run_atom#{atomref_ids[i]}.sh')
#-------------------------------------------------------------------------------       
#===============================================================================
if __name__ == '__main__':
   mpicmd = 'srun'
   
   # Path to the executable pp.x
   exe_path = os.environ['HOME'] + '/lmi-qe-mod/bin'

   prefix = 'MoTe2'
   outdir = './outdir'

   plot_num = 1
   iflag = 0

   latvec_file = 'inp/latvec.dat'
   atoms_file = 'inp/atompos.dat'
   
   inp_prefix = 'inp/'
   out_prefix = 'outfiles/'
   pot_prefx = 'outfiles/'
   
   # Either one goes through all atoms or specify some particular atoms
   atomref_ids = np.arange(6)
   
   pp = ExtractPotential(mpicmd,exe_path,prefix,outdir,
                         plot_num,iflag,latvec_file,atoms_file,
                         inp_prefix,out_prefix,pot_prefx)
   
   pp.run(atomref_ids)