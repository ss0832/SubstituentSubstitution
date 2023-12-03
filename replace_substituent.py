import numpy as np
import sys
import os
import copy
import itertools
from glob import glob
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

rotate_num = 3000
distance_num = 100
threshold = 1.30

def elem_charactor(target_elem_list):
    elements = ["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"]
    element_num = [str(n) for n in range(1,len(elements)+1)]
    for num, elem in enumerate(target_elem_list):
        for e_num in element_num:
            if e_num == elem:
                target_elem_list[num] = elements[int(e_num)-1]
    return target_elem_list


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += np.pi*2
    return r, theta, phi

    
def rotation(a,theta1, theta2):
    
    r = Rotation.from_euler('zy', [theta1, theta2], degrees=False)
    rotated_v = r.apply(a)
    
    return rotated_v



def read_target_molecule(file):
    with open(file,"r") as f:
        words = f.readlines()
        start_data = []
        for word in words:
            start_data.append(word.split())
            

    return start_data

def extract_num(start_data):
    element_list = []
    num_list = []

    for i in range(2, len(start_data)):
        element_list.append(start_data[i][0])
        num_list.append(start_data[i][1:4])
      
    return element_list, num_list
  
def replace(target_data, sub_group_data, replace_elem_list):
    target_elem_list, target_num_list = extract_num(target_data)
    sub_elem_list, sub_num_list = extract_num(sub_group_data)
    sub_vector =  np.array(sub_num_list[1], dtype="float64") - np.array(sub_num_list[0], dtype="float64")#direction of substituent (root→tip)
    print("sub_vector:",sub_vector)
    for replace_elem in replace_elem_list:
        
        replace_vector = np.array(target_num_list[replace_elem[0]-1], dtype="float64") - np.array(target_num_list[replace_elem[1]-1], dtype="float64")#tip→root
        print("replace_vector:",replace_vector)
        
        _, rep_theta1, rep_theta2 = cart2sph(*replace_vector)
        _, sub_theta1, sub_theta2 = cart2sph(*sub_vector)
        new_sub_num_list = []
        new_target_num_list = []
        
        for num in range(len(target_num_list)):
            new_target_num = np.array(target_num_list[num], dtype="float64") - np.array(target_num_list[replace_elem[1]-1], dtype="float64")
            new_target_num_list.append(rotation(new_target_num, -1*rep_theta1, -1*rep_theta2))

        for num in range(len(sub_num_list)):
            new_sub_num = np.array(sub_num_list[num], dtype="float64") - np.array(sub_num_list[0], dtype="float64")
            new_sub_num_list.append(rotation(new_sub_num, -1*sub_theta1, -1*sub_theta2))
        
        target_elem_list[replace_elem[1]-1] = sub_elem_list[0]
        new_target_num_list[replace_elem[1]-1] = new_sub_num_list[0]
        
        for k in range(rotate_num):
            rotate_flag = False
            coords1 = [list(map(float,new_target_num_list[n])) for n in range(len(new_target_num_list)) if (n != replace_elem[0]-1) and (n != replace_elem[1]-1)]#Atoms adjacent to the atom to be substituted are excluded.
            coords2 = [list(map(float,new_sub_num_list[n])) for n in range(len(new_sub_num_list)) if n > 1]
            
            if len(coords2) == 0:
                print("OK")
                break
            coords1_x, coords1_y, coords1_z, coords2_x, coords2_y, coords2_z = [], [], [], [], [], []
            for i, j in itertools.product(coords1,coords2):
                coords1_x.append(i[0])
                coords1_y.append(i[1])
                coords1_z.append(i[2])
                coords2_x.append(j[0])
                coords2_y.append(j[1])
                coords2_z.append(j[2])
            
            dists = np.sqrt((np.array(coords1_x, dtype="float64")-np.array(coords2_x, dtype="float64"))**2+(np.array(coords1_y, dtype="float64")-np.array(coords2_y, dtype="float64"))**2+(np.array(coords1_z, dtype="float64")-np.array(coords2_z, dtype="float64"))**2)
            if np.any(dists < threshold):#threshold  
                rotate_flag = True
                
                     
            if rotate_flag:
                new_sub_num_list = rotate_points_around_vector(new_sub_num_list-new_sub_num_list[0],new_sub_num_list[0]-new_sub_num_list[1],(2*np.pi)/int(rotate_num/distance_num))+new_sub_num_list[0]
            else:
                print("OK")
                break

            if k % int(rotate_num/distance_num) == 0:
                new_sub_num_list = new_sub_num_list.T
                new_sub_num_list[2] -= 0.02
                new_target_num_list[replace_elem[1]-1][2] -= 0.02
                new_sub_num_list = new_sub_num_list.T
        else:
            print("Please check your xyz file.")
        
        for i in range(2,len(sub_elem_list)):
            target_elem_list.append(sub_elem_list[i])
            new_target_num_list.append(new_sub_num_list[i].tolist())
        target_num_list = copy.copy(new_target_num_list)
       
        atom_num = len(target_elem_list)
    
    
    
    target_elem_list = elem_charactor(target_elem_list)        
    
        
    return atom_num, target_elem_list, target_num_list

def rotate_points_around_vector(points, vector, angle):
    # Convert axis of rotation to unit vector
    vector = vector / np.linalg.norm(vector)

    # Calculate Quaternion
    rotation = Rotation.from_rotvec(vector * np.radians(angle))
    quat = rotation.as_quat()

    # Rotate coordinates in quaternions
    rotated_points = Rotation.from_quat(quat).apply(points)

    return rotated_points

def make_xyz_file(atom_num, target_elem_list, target_num_list,group, file):
    with open(file[:-4]+"_"+group[:-4]+".xyz","w") as f:
        f.write(str(atom_num)+"\n")
        f.write("w\n")
        for i in range(len(target_elem_list)):
            target_elem_list[i] = [target_elem_list[i]] 
            target_num_list[i] = list(map(float,target_num_list[i]))
            target_elem_list[i].extend(target_num_list[i])
            f.write("{0:<2}                  {1:>15.12f}   {2:>15.12f}   {3:>15.12f}".format(*target_elem_list[i])+"\n")
            

    return np.array(target_num_list, dtype="float64")
        
def main(file_list, replace_elem_list):#list[file_name,...]. , list[[adjacent atom,atom to be replaced],[adjacent atom,atom to be replaced]...]
    for file in file_list:
        target_data = read_target_molecule(file)
        print("target_molecule:",file)
        substituent_group_list = glob("./substitiuent_group/*.xyz")
        
        for group in substituent_group_list:
            print("substituent_group:",group)
            sub_group_data = read_target_molecule(group)
            atom_num, target_elem_list, target_num_list = replace(target_data,sub_group_data,replace_elem_list)
            make_xyz_file(atom_num, target_elem_list, target_num_list,os.path.basename(group),os.path.basename(file))
            
    
    return

if __name__ == '__main__':

    replace_elem_list = sys.argv[2:]#Adjacent atoms, atoms to be substituted
    if len(replace_elem_list) % 2 == 1:
        print("incorrect input...")
        sys.exit(1)
    replace_elem_list = [list(map(int,replace_elem_list[i:i+2])) for i in range(0, len(replace_elem_list), 2)]
    print("replace_elem_num:",replace_elem_list)
    input_file = sys.argv[1]
    file_list = [input_file]

            
    
    main(file_list,replace_elem_list)

