import numpy as np
import os
import glob
import copy
import math

import argparse

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
            
    return CRL[element]#ang.

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-m", "--metal_atom", type=str, default="1", help='metal atom number', required=True)

    parser.add_argument("-l", "--ligand_donor_atoms", type=str, nargs="*", default="2,3", help='donor atom number of ligand', required=True)
    parser.add_argument("-ls", "--ligand_substituent_donor_atoms", type=str,nargs="*", default="1,2", help='donor atom number of substituent ligand')
    parser.add_argument("-ld", "--substitute_ligand_directory", type=str, default="./ligand_group", help='ligand for substitution')

    
    config = parser.parse_args()
    return config

def number_element(num):
    elem = {1: "H",  2:"He",
         3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 10:"Ne", 
        11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 18:"Ar",
        19:"K", 20:"Ca", 21:"Sc", 22:"Ti", 23:"V", 24:"Cr", 25:"Mn", 26:"Fe", 27:"Co", 28:"Ni", 29:"Cu", 30:"Zn", 31:"Ga", 32:"Ge", 33:"As", 34:"Se", 35:"Br", 36:"Kr",
        37:"Rb", 38:"Sr", 39:"Y", 40:"Zr", 41:"Nb", 42:"Mo",43:"Tc",44:"Ru",45:"Rh", 46:"Pd", 47:"Ag", 48:"Cd", 49:"In", 50:"Sn", 51:"Sb", 52:"Te", 53:"I", 54:"Xe",
        55:"Cs", 56:"Ba", 57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm", 63:"Eu", 64:"Gd", 65:"Tb", 66:"Dy" ,67:"Ho", 68:"Er", 69:"Tm", 70:"Yb", 71:"Lu", 72:"Hf", 73:"Ta", 74:"W", 75:"Re", 76:"Os", 77:"Ir", 78:"Pt", 79:"Au", 80:"Hg", 81:"Tl", 82:"Pb", 83:"Bi", 84:"Po", 85:"At", 86:"Rn"}
        
    return elem[num]



class ReplaceLigand:
    def __init__(self, config):
        def num_parse(numbers):
            sub_list = []
            
            sub_tmp_list = numbers.split(",")
            for sub in sub_tmp_list:                        
                if "-" in sub:
                    for j in range(int(sub.split("-")[0]),int(sub.split("-")[1])+1):
                        sub_list.append(j)
                else:
                    sub_list.append(int(sub))    
            return sub_list
        self.lig_directory = config.substitute_ligand_directory
        tmp_ligand_donor_atoms = []
        for i in range(len(config.ligand_donor_atoms)):
            tmp_ligand_donor_atoms += num_parse(config.ligand_donor_atoms[i])
        self.ligand_donor_atoms = np.array(tmp_ligand_donor_atoms) - 1 
        
        tmp_ligand_substituent_donor_atoms = []
        for i in range(len(config.ligand_substituent_donor_atoms)):
            tmp_ligand_substituent_donor_atoms += num_parse(config.ligand_substituent_donor_atoms[i])
        self.ligand_substituent_donor_atoms = np.array(tmp_ligand_substituent_donor_atoms) - 1 
        
        tmp_metal_atom = []
        for i in range(len(config.metal_atom)):
            tmp_metal_atom += num_parse(config.metal_atom[i])
        self.metal_atom = np.array(tmp_metal_atom) - 1
        
        
        self.ligand_donor_atoms = self.ligand_donor_atoms.tolist()
        self.metal_atom = self.metal_atom.tolist()
        self.ligand_substituent_donor_atoms = self.ligand_substituent_donor_atoms.tolist()

        self.ligand_list = glob.glob(self.lig_directory+"/*.xyz")
        self.covalent_radii_threshold_scale = 1.2
        self.iteration = 500
        self.numerical_derivative = 1e-3
        return
    
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


    def kabsch_algorithm(self, P, Q):
        #scipy.spatial.transform.Rotation.align_vectors
        #P is rotated.
        centroid_P = np.array([[np.mean(P.T[0][self.ligand_substituent_donor_atoms]), np.mean(P.T[1][self.ligand_substituent_donor_atoms]), np.mean(P.T[2][self.ligand_substituent_donor_atoms])]], dtype="float64")
        centroid_Q = np.array([[np.mean(Q.T[0][self.ligand_donor_atoms]), np.mean(Q.T[1][self.ligand_donor_atoms]), np.mean(Q.T[2][self.ligand_donor_atoms])]], dtype="float64")
        P -= centroid_P
        Q -= centroid_Q

        lig_donor_P = copy.copy(P[self.ligand_substituent_donor_atoms])
        lig_donor_Q = copy.copy(Q[self.ligand_donor_atoms])



        H = np.dot(lig_donor_P.T, lig_donor_Q)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = np.dot(Vt.T, U.T)
        
        T = centroid_Q - np.dot(R, centroid_P.T).T
        #P = np.dot(R, P.T).T

        return R, T
        
    def check_atom_connectivity(self, mol_list, element_list, atom_num, metal_atom_num, covalent_radii_threshold_scale=1.2):#atom_num:list, metal_atom_num:list
        connected_atoms = copy.copy(atom_num)
        searched_atoms = copy.copy(metal_atom_num)
        
        while True:
            for i in connected_atoms:
                if i in searched_atoms:
                    continue
              
                for j in range(len(mol_list)):
                    dist = np.linalg.norm(np.array(mol_list[i], dtype="float64") - np.array(mol_list[j], dtype="float64"))
                    
                    covalent_dist_threshold = covalent_radii_threshold_scale * (covalent_radii_lib(element_list[i]) + covalent_radii_lib(element_list[j]))
                    
                    if dist < covalent_dist_threshold:
                        if not j in connected_atoms:
                            connected_atoms.append(j)
                
                searched_atoms.append(i)
            
            if len(connected_atoms) == len(searched_atoms):
                break
     
        return sorted(connected_atoms)
    
    

    
    def main(self, file):
        base_mol_elem_list, base_mol_coord = self.xyz2list(file)

        base_lig_mol_num_list = self.check_atom_connectivity(base_mol_coord, base_mol_elem_list, self.ligand_donor_atoms, self.metal_atom, self.covalent_radii_threshold_scale)
        
        base_mol_lig_coord = base_mol_coord[base_lig_mol_num_list]
        
        base_mol_substrate_coord = np.array([base_mol_coord[i] for i in range(len(base_mol_elem_list)) if not i in base_lig_mol_num_list])
        
        

        base_mol_substrate_elem_list = [base_mol_elem_list[i] for i in range(len(base_mol_elem_list)) if not i in base_lig_mol_num_list]

        base_mean_donor_atom_coord = np.array([0.0, 0.0, 0.0], dtype="float64")
        for j in range(len(self.ligand_donor_atoms)):
            base_mean_donor_atom_coord += base_mol_coord[j]
        base_mean_donor_atom_coord /= len(base_mean_donor_atom_coord)

        for lig_file in self.ligand_list:
            print(lig_file)
            lig_mol_elem_list, lig_mol_coord = self.xyz2list(lig_file)

            P = copy.copy(lig_mol_coord)
            Q = copy.copy(base_mol_lig_coord)
            centroid_Q = np.array([np.mean(Q.T[0][self.ligand_donor_atoms]), np.mean(Q.T[1][self.ligand_donor_atoms]), np.mean(Q.T[2][self.ligand_donor_atoms])], dtype="float64")
            centroid_P = np.array([np.mean(P.T[0][self.ligand_substituent_donor_atoms]), np.mean(P.T[1][self.ligand_substituent_donor_atoms]), np.mean(P.T[2][self.ligand_substituent_donor_atoms])], dtype="float64")
            R, T = self.kabsch_algorithm(P, Q)
            
            P_t = np.dot(R, P.T).T + T
          
            rot_lig_mol_coord = P_t

            new_mol_coord = np.concatenate([base_mol_coord[self.metal_atom], base_mol_substrate_coord, rot_lig_mol_coord])
            new_elem_list = [base_mol_elem_list[j] for j in self.metal_atom] + base_mol_substrate_elem_list + lig_mol_elem_list

            self.list2xyz(file, new_elem_list, new_mol_coord, os.path.splitext(os.path.basename(lig_file))[0])

        return
    
if __name__ == '__main__':
    config = argparser()
    RL = ReplaceLigand(config)
    job_name = config.INPUT
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
        RL.main(file)
        print()
    print("all complete...")