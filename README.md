# SubstituentSubstitution
Substituent Substitution Program (for .xyz file format)

A program that saves .xyz files in which any specified end atom of a specified xyz file is converted to a substituent in the substituent_group (folder).

You can also create your own substituents. If you make one, it must be in .xyz file format and the second atom must be a dummy atom X. 

The first atom should be adjacent to the dummy atom X. (It does not have to be a dummy atom, but it is recommended that it be a dummy atom for the sake of convenience.)

Do not change the name of the "substance_group" folder (the program will not work if you do not change the code).

A python3 runtime environment is required.

### Required modules
 - scipy
 - numpy


### How to use

`[python] [replace_substituents.py] [xxx.xyz (.xyz file)] [(atoms adjacent to atom to be replaced) (terminal atom to be replaced) ...] `

### Example
```
python replace_substituent.py SEG_PHOS.xyz 2 23 2 24 20 25 20 26
```

### License

The license of this program is MIT license.

