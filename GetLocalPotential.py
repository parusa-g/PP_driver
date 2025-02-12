import os 
import numpy as np
import f90nml
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
   def WriteRunScript(self,atomref_ids):
      
      run_script = 'run_pp.sh'

      FileInp_init = self.inp_prefix + self.prefix + '.pp.inp'
      FileOut_init = self.out_prefix + self.prefix + '.pp.out'

      with open(run_script,'w') as f:
         f.write('#!/bin/bash\n')
         f.write(f'#SBATCH --job-name=pp.x\n')
         f.write('#SBATCH --cluster=merlin6\n')
         f.write('#SBATCH --partition=hourly\n')
         f.write('#SBATCH --time=00:05:00\n')
         f.write('#SBATCH --output=slurm/%j.out\n')
         f.write('#SBATCH --error=slurm/%j.err\n')
         f.write('#SBATCH --hint=nomultithread\n')
         f.write('#SBATCH --ntasks=6\n')
         f.write('#SBATCH --ntasks-per-core=1\n')
         f.write('#SBATCH --nodes=1\n')
         
         atomref_ids_str = ','.join([str(i) for i in atomref_ids])
         f.write(f'#SBATCH --array=[{atomref_ids_str}]\n')
         f.write('\n')
         
         f.write(f'PP={self.exe}\n')
         f.write('\n')
         
         f.write(f'if [ ! -f {FileOut_init} ]; then\n')
         f.write(f'  {self.mpicmd} $PP < {FileInp_init} > {FileOut_init}\n')
         f.write(f'fi\n')
         f.write('\n')
         f.write(f'while ! grep -q "JOB DONE" {FileOut_init}; do\n')
         f.write(f'  sleep 1\n')
         f.write('done\n')
         f.write('\n')
         
         label = 'Atom\#${SLURM_ARRAY_TASK_ID}'
         file_in = self.inp_prefix + self.prefix + '_' + label + '.pp.inp'
         file_out = self.out_prefix + self.prefix + '_' + label + '.pp.out'
         
         f.write(f'{self.mpicmd} $PP < {file_in} > {file_out}\n')
         f.write('\n')

#-------------------------------------------------------------------------------
   def WriteInputs(self,atomref_ids):
      
      FileInp_init = self.inp_prefix + self.prefix + '.pp.inp'
      self.WriteInputPPfile(FileInp_init, initialized=False)
      
      for i in range(len(atomref_ids)):
         atomref_id = atomref_ids[i]

         atom_label = 'Atom#' + str(atomref_id)

         nn_id, nnvec = self.FindNearestNeighborVector(atomref_id)

         print(f'Prepare an input for atom # {atomref_id}: {self.atom_labels[atomref_id]}')
         print(f'Nearest neighbor atom: {self.atom_labels[nn_id]}, # {nn_id}')
         print(f'Nearest neighbor vector: {nnvec[0]:.6f}, {nnvec[1]:.6f}, {nnvec[2]:.6f}')

         vec = 0.5 * nnvec
         
         file_inp = self.inp_prefix + self.prefix + '_' + atom_label + '.pp.inp'

         self.WriteInputPPfile(file_inp, atom_label, self.atompos_cart[atomref_id,:], vec)
         
#-------------------------------------------------------------------------------
   def run(self,atomref_ids):
      self.WriteInputs(atomref_ids)
      self.WriteRunScript(atomref_ids)
      
      os.system('sbatch run_pp.sh')
      
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