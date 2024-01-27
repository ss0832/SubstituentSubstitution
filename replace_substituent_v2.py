from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import glob
import copy
import math



def element_number(elem):
    num = {"H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, 
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,"Tc": 43,"Ru": 44,"Rh": 45,"Pd": 46,"Ag": 47,"Cd": 48,"In": 49,"Sn": 50,"Sb": 51,"Te": 52,"I": 53,"Xe": 54,
        "Cs": 55 ,"Ba": 56, "La": 57,"Ce":58,"Pr": 59,"Nd": 60,"Pm": 61,"Sm": 62,"Eu": 63,"Gd": 64,"Tb": 65,"Dy": 66,"Ho": 67,"Er": 68,"Tm": 69,"Yb": 70,"Lu": 71,"Hf": 72,"Ta": 73,"W": 74,"Re": 75,"Os": 76,"Ir": 77,"Pt": 78,"Au": 79,"Hg": 80,"Tl": 81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86}
        
    return num[elem]

def number_element(num):
    elem = {1: "H",  2:"He",
         3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
        11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
        19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr",
        37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",43:"Tc",44:"Ru",45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe",
        55:"Cs", 56:"Ba", 57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy" ,67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn"}
        
    return elem[num]

def UFF_effective_charge_lib(element):
    if element is int:
        element = number_element(element)
    UFF_EC = {'H':0.712,'He': 0.098,
                        'Li' : 1.026 ,'Be': 1.565, 'B': 1.755,'C': 1.912, 'N': 2.544,'O': 2.300, 'F': 1.735,'Ne': 0.194, 
                        'Na': 1.081, 'Mg': 1.787,'Al': 1.792,'Si': 2.323, 'P': 2.863, 'S': 2.703,'Cl': 2.348,'Ar': 0.300,
                        'K': 1.165 ,'Ca': 2.141,'Sc': 2.592,'Ti': 2.659,'V': 2.679, 'Cr': 2.463,'Mn': 2.430, 'Fe': 2.430,'Co': 2.430 ,'Ni': 2.430 ,'Cu': 1.756 ,'Zn': 1.308,'Ga': 1.821, 'Ge': 2.789,'As': 2.864,'Se': 2.764,'Br': 2.519,'Kr': 0.452,
                        'Rb': 1.592,'Sr': 2.449,'Y': 3.257,'Zr': 3.667,'Nb': 3.618,'Mo': 3.400, 'Tc': 3.400,'Ru': 3.400,'Rh': 3.508, 'Pd': 3.210,'Ag': 1.956,'Cd': 1.650,'In': 2.070,'Sn': 2.961,'Sb': 2.704,'Te': 2.882, 'I': 2.650, 'Xe': 0.556, 
                        'Cs': 1.573,'Ba': 2.727, 'La': 3.300, 'Ce': 3.300,'Pr': 3.300,'Nd':3.300,'Pm':3.300,'Sm':3.300,'Eu':3.300,'Gd':3.300,'Tb':3.300,'Dy':3.300,'Ho': 3.416 ,'Er': 3.300,'Tm': 3.300,'Yb': 2.618,'Lu': 3.271,'Hf': 3.921,
                        'Ta': 4.075,'W': 3.70,'Re': 3.70,'Os': 3.70,'Ir': 3.731,'Pt': 3.382,'Au': 2.625,'Hg': 1.750,'Tl': 2.068,'Pb': 2.846,'Bi': 2.470,'Po': 2.330,'At': 2.240,'Rn': 0.583}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #charge
    return UFF_EC[element]

def UFF_VDW_distance_lib(element):
    if element is int:
        element = number_element(element)
    UFF_VDW_distance = {'H':2.886,'He':2.362 ,
                        'Li' : 2.451 ,'Be': 2.745, 'B':4.083 ,'C': 3.851, 'N':3.660,'O':3.500 , 'F':3.364,'Ne': 3.243, 
                        'Na':2.983,'Mg': 3.021 ,'Al':4.499 ,'Si': 4.295, 'P':4.147, 'S':4.035 ,'Cl':3.947,'Ar':3.868 ,
                        'K':3.812 ,'Ca':3.399 ,'Sc':3.295 ,'Ti':3.175 ,'V': 3.144, 'Cr':3.023 ,'Mn': 2.961, 'Fe': 2.912,'Co':2.872 ,'Ni':2.834 ,'Cu':3.495 ,'Zn':2.763 ,'Ga': 4.383,'Ge':4.280,'As':4.230 ,'Se':4.205,'Br':4.189,'Kr':4.141 ,
                        'Rb':4.114 ,'Sr': 3.641,'Y':3.345 ,'Zr':3.124 ,'Nb':3.165 ,'Mo':3.052 ,'Tc':2.998 ,'Ru':2.963 ,'Rh':2.929 ,'Pd':2.899 ,'Ag':3.148 ,'Cd':2.848 ,'In':4.463 ,'Sn':4.392 ,'Sb':4.420 ,'Te':4.470 , 'I':4.50, 'Xe':4.404 , 
                        'Cs':4.517 ,'Ba':3.703 , 'La':3.522 , 'Ce':3.556 ,'Pr':3.606 ,'Nd':3.575 ,'Pm':3.547 ,'Sm':3.520 ,'Eu':3.493 ,'Gd':3.368 ,'Tb':3.451 ,'Dy':3.428 ,'Ho':3.409 ,'Er':3.391 ,'Tm':3.374 ,'Yb':3.355,'Lu':3.640 ,'Hf': 3.141,
                        'Ta':3.170 ,'W':3.069 ,'Re':2.954 ,'Os':3.120 ,'Ir':2.840 ,'Pt':2.754 ,'Au':3.293 ,'Hg':2.705 ,'Tl':4.347 ,'Pb':4.297 ,'Bi':4.370 ,'Po':4.709 ,'At':4.750 ,'Rn': 4.765}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 #ang.
                
    return UFF_VDW_distance[element] / UnitValueLib().bohr2angstroms#Bohr

def UFF_VDW_well_depth_lib(element):
    if element is int:
        element = number_element(element)         
    UFF_VDW_well_depth = {'H':0.044, 'He':0.056 ,
                          'Li':0.025 ,'Be':0.085 ,'B':0.180,'C': 0.105, 'N':0.069, 'O':0.060,'F':0.050,'Ne':0.042 , 
                          'Na':0.030, 'Mg':0.111 ,'Al':0.505 ,'Si': 0.402, 'P':0.305, 'S':0.274, 'Cl':0.227,  'Ar':0.185 ,
                          'K':0.035 ,'Ca':0.238 ,'Sc':0.019 ,'Ti':0.017 ,'V':0.016 , 'Cr':0.015, 'Mn':0.013 ,'Fe': 0.013,'Co':0.014 ,'Ni':0.015 ,'Cu':0.005 ,'Zn':0.124 ,'Ga':0.415 ,'Ge':0.379, 'As':0.309 ,'Se':0.291,'Br':0.251,'Kr':0.220 ,
                          'Rb':0.04 ,'Sr':0.235 ,'Y':0.072 ,'Zr':0.069 ,'Nb':0.059 ,'Mo':0.056 ,'Tc':0.048 ,'Ru':0.056 ,'Rh':0.053 ,'Pd':0.048 ,'Ag':0.036 ,'Cd':0.228 ,'In':0.599 ,'Sn':0.567 ,'Sb':0.449 ,'Te':0.398 , 'I':0.339,'Xe':0.332 , 
                          'Cs':0.045 ,'Ba':0.364 , 'La':0.017 , 'Ce':0.013 ,'Pr':0.010 ,'Nd':0.010 ,'Pm':0.009 ,'Sm':0.008 ,'Eu':0.008 ,'Gd':0.009 ,'Tb':0.007 ,'Dy':0.007 ,'Ho':0.007 ,'Er':0.007 ,'Tm':0.006 ,'Yb':0.228 ,'Lu':0.041 ,'Hf':0.072 ,
                          'Ta':0.081 ,'W':0.067 ,'Re':0.066 ,'Os':0.037 ,'Ir':0.073 ,'Pt':0.080 ,'Au':0.039 ,'Hg':0.385 ,'Tl':0.680 ,'Pb':0.663 ,'Bi':0.518 ,'Po':0.325 ,'At':0.284 ,'Rn':0.248, 'X':0.010}#H...Rn J. Am. Chem. Soc., 1992, 114, 10024 # kcal/mol
                
    return UFF_VDW_well_depth[element] / UnitValueLib().hartree2kcalmol #hartree
                

def covalent_radii_lib(element):
    if element is int:
        element = number_element(element)
    CRL = {"H": 0.32, "He": 0.46, 
           "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, 
           "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, 
           "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, 
           "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,
           "Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42, 'X':1.000}#ang.
    # ref. Pekka Pyykkö; Michiko Atsumi (2009). “Molecular single-bond covalent radii for elements 1 - 118”. Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. (H...Rn)
            
    return CRL[element] / UnitValueLib().bohr2angstroms#Bohr


class UnitValueLib: 
    def __init__(self):
        self.hartree2kcalmol = 627.509 #
        self.bohr2angstroms = 0.52917721067 #
        self.hartree2kjmol = 2625.500 #
        return

class ReplaceSubstituentGroup:
    def __init__(self):
        self.rand_search_iter = 300
        self.opt_max_iter = 100
        self.threshold = 10.0
        self.DELTA = 1e-4
        self.bohr2angstroms = UnitValueLib().bohr2angstroms
        self.hartree2kcalmol = UnitValueLib().hartree2kcalmol
        self.rand_dist_flag = True
        self.opt_flag = True
        return

    def initialization(self, file, replace_atoms):
        self.element_list, self.coord = self.xyz2list(file)
        self.replace_atoms = np.array(replace_atoms, dtype="int32") - 1
        self.atom_pairs = self.neighbor_atom_check(self.coord, self.replace_atoms)
        self.substituent_list = glob.glob("./substitiuent_group/*.xyz")

    def xyz2list(self, file_path):

        with open(file_path, "r") as f:
            words = f.read().splitlines()
        
        element_list = []
        coord = []

        for word in words[2:]:
            splited_word = word.split()
            element_list.append(splited_word[0])
            coord.append(np.array(splited_word[1:4], dtype="float64"))
        coord = np.array(coord, dtype="float64")
        return element_list, coord

    def list2xyz(self, file_path, element_list, coords, substituent_name):
        natoms = len(element_list)

        with open(file_path[:-4]+"_"+substituent_name+".xyz" ,"w") as f: 
            f.write(str(natoms)+"\n\n")
            for i, coord in enumerate(coords):
                str_coord = [element_list[i]] + coord.tolist()

                f.write("{0:<2}                  {1:>15.12f}   {2:>15.12f}   {3:>15.12f}".format(*str_coord)+"\n")
                

        return


    def neighbor_atom_check(self, coord, replace_atoms):
        atom_pairs = []

        for atom in replace_atoms:
            min_score = 1e+10
            neighbor_atom_num = 0
            for num, c in enumerate(coord):
                if num == atom:
                    continue
                distance = np.linalg.norm(c - coord[atom])
                if min_score > distance:
                    min_score = distance
                    neighbor_atom_num = num
            
            atom_pairs.append([neighbor_atom_num, atom])#root_atom, replace_atom

        #print(atom_pairs)
        return atom_pairs
            

    def rotate_points_around_vector(self, points, vector, angle):#angle:rad.
        vector = vector / np.linalg.norm(vector)
        rotation = R.from_rotvec(vector * angle)
        quat = rotation.as_quat()
        rotated_points = R.from_quat(quat).apply(points)
        return rotated_points

    def LJ_potential(self, element_list, coord, sub_atoms):
        energy = 0.0
        for num, atoms in enumerate(sub_atoms):
            for atom in atoms:
                for i in range(len(coord)):
                    if i in atoms or i in self.atom_pairs[num]:
                        continue

                    distance = np.linalg.norm(coord[i] - coord[atom])
                    if distance > 15.0:#cutoff
                        continue
                    eps_1 = UFF_VDW_well_depth_lib(element_list[i])
                    eps_2 = UFF_VDW_well_depth_lib(element_list[atom])
                    sigma_1 = UFF_VDW_distance_lib(element_list[i])
                    sigma_2 = UFF_VDW_distance_lib(element_list[atom])
                    eps = math.sqrt(eps_1 * eps_2)
                    sigma = math.sqrt(sigma_1 * sigma_2)

                    energy +=  eps * ((sigma/distance)**12 -2 * (sigma/distance)**6)

        return energy

    def elecstatic_potential(self, element_list, coord, sub_atoms):
        energy = 0.0
        epsilon = 1.0
        for num, atoms in enumerate(sub_atoms):
            for atom in atoms:
                for i in range(len(coord)):
                    if i in atoms or i in self.atom_pairs[num]:
                        continue
                    distance = np.linalg.norm(coord[i] - coord[atom])
                    if distance > 30.0:#cutoff
                        continue
                    electrostaticcharge = UFF_effective_charge_lib(element_list[i]) * UFF_effective_charge_lib(element_list[atom])
                    
                    energy += ((332.0637 * electrostaticcharge) / (epsilon * distance)) / self.hartree2kcalmol * self.bohr2angstroms
        return energy

    def atom_struct_randomizer(self, coord, element_list, atom_pairs):
        valiables = [] #angle, dist (number of atom pairs * 2)
        rng = np.random.default_rng()
        for pairs in atom_pairs:
            cov_dist = covalent_radii_lib(element_list[pairs[0]]) + covalent_radii_lib(element_list[pairs[1]])#bohr
            if self.rand_dist_flag:
                rand_dist = rng.uniform(cov_dist * 0.95, cov_dist * 1.1)
            else:
                rand_dist = np.linalg.norm(coord[pairs[1]] - coord[pairs[0]]) / self.bohr2angstroms
            rand_angle = rng.uniform(0.0, 2*np.pi)
            valiables.extend([rand_angle, rand_dist])

        return valiables

    def atom_struct_changer(self, coord, sub_atom_list, atom_pairs, valiables):
        
        for i in range(int(len(valiables)/2)):
            length = valiables[2*i+1] * self.bohr2angstroms
            angle = valiables[2*i]
            vector = coord[atom_pairs[i][1]] - coord[atom_pairs[i][0]]
            rot, _ = R.align_vectors([[0, 0, 1]], [vector])
            tmp_coord = rot.apply(coord[sub_atom_list[i]] - coord[atom_pairs[i][0]])

            tmp_rot_coord = self.rotate_points_around_vector(tmp_coord, np.array([0, 0, 1]), angle)
            rot, _ = R.align_vectors([vector], [[0, 0, 1]])
            rot_coord = rot.apply(tmp_rot_coord)
           
            prev_coord = copy.copy(coord)
            coord[sub_atom_list[i]] = copy.copy(rot_coord + coord[atom_pairs[i][0]])
            
            
            coord[[atom_pairs[i][1]] + sub_atom_list[i]] = copy.copy(coord[[atom_pairs[i][1]] + sub_atom_list[i]] - vector + length * vector/np.linalg.norm(vector))
            

        return coord
        
    def FSB_hessian_update(self, hess, displacement, delta_grad):
        """
        FSB
        J. Chem. Phys. 1999, 111, 10806
        """
        A = delta_grad - np.dot(hess, displacement)
        delta_hess_SR1 = np.dot(A, A.T) / np.dot(A.T, displacement) 
        delta_hess_BFGS = (np.dot(delta_grad, delta_grad.T) / np.dot(displacement.T, delta_grad)) - (np.dot(np.dot(np.dot(hess, displacement) , displacement.T), hess.T)/ np.dot(np.dot(displacement.T, hess), displacement))
        Bofill_const = np.dot(np.dot(np.dot(A.T, displacement), A.T), displacement) / np.dot(np.dot(np.dot(A.T, A), displacement.T), displacement)
        delta_hess = np.sqrt(Bofill_const)*delta_hess_SR1 + (1 - np.sqrt(Bofill_const))*delta_hess_BFGS

        return delta_hess
        
    def FSB_quasi_newton_method(self, geom_num_list, pre_geom, pre_g, g):
        
        delta_grad = g - pre_g
        displacement = geom_num_list - pre_geom
        
        delta_hess = self.FSB_hessian_update(self.model_hess, displacement, delta_grad)
        
        new_hess = self.model_hess + delta_hess 
        
        DELTA_for_QNM = self.DELTA
        
        try:
            move_vector = DELTA_for_QNM*np.dot(np.linalg.inv(new_hess), g)
        except:
            move_vector = g / np.linalg.norm(g) * min(0.1, np.linalg.norm(g))
        
        self.model_hess = new_hess
        return move_vector
        
        
    def optimizer(self, coord, element_list, sub_atom_list):
        derivative_delta = 1e-4
        min_energy_score = 1e+10
        min_valiables = []
        self.model_hess = np.eye(len(sub_atom_list)*2)

        for i in range(self.rand_search_iter): 

            valiables = self.atom_struct_randomizer(coord, element_list, self.atom_pairs)
            
            coord = self.atom_struct_changer(coord, sub_atom_list, self.atom_pairs, valiables)
            
            
            candidate_ene = self.LJ_potential(element_list, coord / self.bohr2angstroms, sub_atom_list) + self.elecstatic_potential(element_list, coord / self.bohr2angstroms, sub_atom_list)
            if candidate_ene < min_energy_score:
                min_energy_score = candidate_ene
                min_valiables = valiables
                min_coord = coord
                print("min_energy (LJ+electrostatic energy): ", min_energy_score)
                print("valiables (angle(rad.), distance(Bohr) ...): ", min_valiables)
                
        min_valiables = np.array(min_valiables, dtype="float64")
        
        cov_dist_list = []
        for pairs in self.atom_pairs:
            cov_dist = covalent_radii_lib(element_list[pairs[0]]) + covalent_radii_lib(element_list[pairs[1]])#bohr
            cov_dist_list.append(cov_dist)
        
        if self.opt_flag:
            print("optimize molecule structure...")
            for i in range(self.opt_max_iter):
                print("# ITR. "+str(i))
                grad_list = []
                for j in range(len(min_valiables)):
                    p_valiables = copy.copy(min_valiables)
                    p_valiables[j] += derivative_delta
                    
                    m_valiables = copy.copy(min_valiables)
                    m_valiables[j] -= derivative_delta
                 
                    p_coord = copy.copy(coord)
                    m_coord = copy.copy(coord)
                    p_coord = self.atom_struct_changer(p_coord, sub_atom_list, self.atom_pairs, p_valiables)
                    m_coord = self.atom_struct_changer(m_coord, sub_atom_list, self.atom_pairs, m_valiables)
                   
                    p_ene = self.LJ_potential(element_list, p_coord / self.bohr2angstroms, sub_atom_list) + self.elecstatic_potential(element_list, p_coord / self.bohr2angstroms, sub_atom_list)
                    m_ene = self.LJ_potential(element_list, m_coord / self.bohr2angstroms, sub_atom_list) + self.elecstatic_potential(element_list, m_coord / self.bohr2angstroms, sub_atom_list)
                
                    grad = (p_ene - m_ene) / (2 * derivative_delta)
                    grad_list.append(grad)
                
                grad_list = np.array(grad_list, dtype="float64")
                print("gradient : ", grad_list)
                if i == 0:
                    prev_valiables = copy.copy(min_valiables)
                    
                    if not self.rand_dist_flag:
                        for k in range(int(len(grad_list)/2)):
                            min_valiables[2*k] -= grad_list[2*k] / np.linalg.norm(grad_list) * 1e-5
                           
                    else:
                        min_valiables -= grad_list / np.linalg.norm(grad_list) * 1e-5
                else:
                    move_vector = self.FSB_quasi_newton_method(min_valiables, prev_valiables, prev_grad_list, grad_list)
                    
                    step_radii = min(np.linalg.norm(move_vector), 0.1)
                    
                    prev_valiables = copy.copy(min_valiables)
                    
                    
                    if not self.rand_dist_flag:
                        for k in range(int(len(grad_list)/2)):
                            min_valiables[2*k] -= step_radii * move_vector[2*k] / np.linalg.norm(move_vector)
                           
                    else:
                        min_valiables -= step_radii * move_vector / np.linalg.norm(move_vector)
                 
                prev_grad_list = grad_list
                print("valiables (angle(rad.), distance(Bohr) ...): ", min_valiables)
                print("grad : |"+str(np.linalg.norm(grad_list))+"|")
                
                
                if self.rand_dist_flag:
                    for k in range(int(len(grad_list)/2)):
                        
                        min_valiables[2*k+1] = np.clip(min_valiables[2*k+1], cov_dist_list[k]*0.95, cov_dist_list[k]*1.1)
                else:
                    pass
                    
                
                
                if np.linalg.norm(grad_list) < self.threshold:
                    print("converged!!!")
                    break
                #raise
            min_coord = self.atom_struct_changer(coord, sub_atom_list, self.atom_pairs, min_valiables)
       
        return min_coord





    def main(self):
        

        for sub_file_path in self.substituent_list:

            print(sub_file_path)
            coord = copy.copy(self.coord)
            element_list = copy.copy(self.element_list)
            
            sub_element_list, sub_coord = self.xyz2list(sub_file_path)
            sub_vector = sub_coord[0] - sub_coord[1]
            sub_atom_num = len(sub_coord) - 2
            origin_molecule_atom_num = len(coord)

            for pair in self.atom_pairs:
                main_molecule_vector = coord[pair[1]] - coord[pair[0]]
                rot, _ = R.align_vectors([main_molecule_vector], [sub_vector])
                base_point_sub_coord = sub_coord - sub_coord[1]
                after_rot_base_point_sub_coord = rot.apply(base_point_sub_coord)
                after_rot_sub_coord = after_rot_base_point_sub_coord + coord[pair[0]] 
                after_rot_sub_coord = after_rot_sub_coord + (after_rot_sub_coord[1] - after_rot_sub_coord[0]) + main_molecule_vector
                element_list[pair[1]] = sub_element_list[0]
                coord = np.block([[coord], [after_rot_sub_coord[2:]]])
                element_list += sub_element_list[2:] 
            
            sub_atom_list = []
            for i in range(len(self.atom_pairs)):
                sub_atom_list.append([j for j in range(origin_molecule_atom_num + i*sub_atom_num, origin_molecule_atom_num + (i+1)*sub_atom_num)])
            if len(sub_element_list) > 2: 
                optimized_coord = self.optimizer(coord, element_list, sub_atom_list)
            else:
                optimized_coord = coord
            self.list2xyz(file, element_list, optimized_coord, os.path.basename(sub_file_path)[:-4])
            print("\nsturcture saved...\n")
            
                

        
        print("complete...")
        return



if __name__ == '__main__':
    import sys
    job_name = sys.argv[1]
    dist_fix_flag = sys.argv[2]
    replace_atoms = sys.argv[3:]

    if "*" in str(job_name):
            
        file_list = []
        for job in job_name:
            print(job)
            tmp = glob(job)
            file_list.extend(tmp)
    else:
        file_list = [job_name]

    for file in file_list:
        print("input file: ", file)
        RSG = ReplaceSubstituentGroup()
        RSG.initialization(file, replace_atoms)
        if dist_fix_flag == "t":
            RSG.rand_dist_flag = False
        RSG.main()
        print()
    print("all complete...")
 



