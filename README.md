# SubstituentSubstitution
Substituent Substitution Program (for .xyz file format)

A program that saves .xyz files in which any specified end atom of a specified xyz file is converted to a substituent in the substituent_group (folder).

You can also create your own substituents. If you make one, it must be in .xyz file format and the second atom must be a dummy atom X. 

The first atom should be adjacent to the dummy atom X. (It does not have to be a dummy atom, but it is recommended that it be a dummy atom for the sake of convenience.)

Do not change the name of the "substance_group" folder (the program will not work if you do not change the code).

A python3 runtime environment is required.

replace_ligand.py is a program that performs ligand replacement.

### Required modules
 - scipy
 - numpy


### How to use

`[python] [replace_substituents.py] [xxx.xyz (.xyz file)] [(atoms adjacent to atom to be replaced) (terminal atom to be replaced) ...] `

`[python] [replace_substituents_v2.py] [xxx.xyz (.xyz file)] [t (if you want to optimize atom distance, please input a word except 't'.)] [(terminal atom to be replaced) ...] `

### Example
```
python replace_substituent.py SEG_PHOS.xyz 2 23 2 24 20 25 20 26
```

```
python replace_substituent_v2.py SEG_PHOS.xyz f 23 24 25 26
```

### TODO

- coding this code in C++ language.
- Improvement of the algorithm of replace_ligand.py (I can't replace different kinds of ligands well. Currently, the structure output by xTB and others must be coarsely optimized to be usable).



### License

The license of this program is MIT license.

