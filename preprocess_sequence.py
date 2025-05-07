"""
Preprocess the RNA sequence to get the QUBO coefficients and sets
Here we use the paper to understand how to preprocess each term
"""
import numpy as np

# Turner energy constants
STACK_ENERGY = {
    ('G','C','C','G'): -3.4,   # outer GC, inner CG
    ('C','G','G','C'): -2.9,
    ('G','C','G','C'): -3.3,
    ('C','G','C','G'): -3.3,
    ('G','C','U','G'): -2.5,
    ('C','G','G','U'): -2.1,
    ('G','U','C','G'): -2.1,
    ('U','G','G','C'): -2.5,
    ('G','U','U','G'): -1.3,
    ('U','G','G','U'): -1.3,
    ('A','U','U','A'): -1.1,
    ('U','A','A','U'): -1.1,
    ('A','U','G','C'): -2.1,
    ('U','A','C','G'): -2.1,
    ('G','C','A','U'): -2.1,
    ('C','G','U','A'): -2.1
}

LOOP_PENALTY = {3: 5.4, 4: 4.1, 5: 3.4, 6: 3.0}  # kcal/mol

def bp_score(b1, b2):
    """
    Calculate base pair score using Turner energy values
    
    Args:
        b1 (str): First base
        b2 (str): Second base
        
    Returns:
        float: Energy score in kcal/mol
    """
    if (b1,b2) in [('G','C'),('C','G')]: return -3.4   # kcal/mol
    if (b1,b2) in [('A','U'),('U','A')]: return -2.1
    if (b1,b2) in [('G','U'),('U','G')]: return -2.3
    return 0


def actual_stems(seq_ss, seq_ps, subdirectory):
    """
    Read in .ct file and give a list of known structure stems
    
    Args:
        seq_ss (str): Path to the .ct file
        seq_ps (str): Path to the .fasta file
        subdirectory (str): Directory containing the files
        
    Returns:
        list: List of actual stems in the structure
    """
    with open(subdirectory+"/"+seq_ss) as file:
        lines = file.readlines()
    
    with open(subdirectory+"/"+seq_ps) as file:
        fasta_lines = file.readlines()
    
    rna = fasta_lines[1]
    
    stems_actual = []
    sip = False                       # stem in progress?
    sl = 0                            # stem length
    last_line = [0, 0, 0, 0, 0, 0]    # initiate last line

    for i in range(0, len(lines)):
        line = lines[i].strip().split()
        if (int(line[4]) != 0 and sip == False):
            sip = True
            temp = [int(line[0]), int(line[4])]
            if (rna[i] == ('G' or 'g') and rna[int(line[4])-1] == ('C' or 'c')) or (rna[i] == ('C' or 'c') and rna[int(line[4])-1] == ('G' or 'g')):
                sl += 3
            if (rna[i] == ('G' or 'g') and rna[int(line[4])-1] == ('U' or 'u')) or (rna[i] == ('U' or 'u') and rna[int(line[4])-1] == ('G' or 'g')) or (rna[i] == ('A' or 'a') and rna[int(line[4])-1] == ('U' or 'u')) or (rna[i] == ('U' or 'u') and rna[int(line[4])-1] == ('A' or 'a')):
                sl += 2
        if (int(line[4]) != 0 and sip == True and (int(last_line[4])-int(line[4]) == 1)):
            if (rna[i] == ('G' or 'g') and rna[int(line[4])-1] == ('C' or 'c')) or (rna[i] == ('C' or 'c') and rna[int(line[4])-1] == ('G' or 'g')):
                sl += 3
            if (rna[i] == ('G' or 'g') and rna[int(line[4])-1] == ('U' or 'u')) or (rna[i] == ('U' or 'u') and rna[int(line[4])-1] == ('G' or 'g')) or (rna[i] == ('A' or 'a') and rna[int(line[4])-1] == ('U' or 'u')) or (rna[i] == ('U' or 'u') and rna[int(line[4])-1] == ('A' or 'a')):
                sl += 2
        if (int(line[4]) == 0 and sip == True):
            sip = False
            temp.append(sl)
            if temp[1] > temp[0]:
                stems_actual.append(temp)
            sl = 0
        if ((int(last_line[4])-int(line[4]) != 1) and int(last_line[4]) != 0  and sip == True):
            temp.append(sl)
            if temp[1] > temp[0]:
                stems_actual.append(temp)
            temp = [int(line[0]), int(line[4])]
            sl = 0
            if (rna[i] == ('G' or 'g') and rna[int(line[4])-1] == ('C' or 'c')) or (rna[i] == ('C' or 'c') and rna[int(line[4])-1] == ('G' or 'g')):
                sl = 3
            if (rna[i] == ('G' or 'g') and rna[int(line[4])-1] == ('U' or 'u')) or (rna[i] == ('U' or 'u') and rna[int(line[4])-1] == ('G' or 'g')) or (rna[i] == ('A' or 'a') and rna[int(line[4])-1] == ('U' or 'u')) or (rna[i] == ('U' or 'u') and rna[int(line[4])-1] == ('A' or 'a')):
                sl = 2
        
        last_line = line
        
    return stems_actual


def potential_stems(seq_ps, subdirectory):
    """
    Read in .fasta file and generate list of potential stems at least 3 base-pairs long
    
    Args:
        seq_ps (str): Path to the .fasta file
        subdirectory (str): Directory containing the file
        
    Returns:
        list: List containing [potential stems, maximum stem energy, RNA sequence, sequence length]
    """
    with open(subdirectory+"/"+seq_ps) as file:
        lines = file.readlines()
    
    rna = lines[1]
    
    matrix = np.zeros((len(rna),len(rna)))
    for diag in range(0, len(matrix)):
        for row in range(0, len(matrix)-diag):
            col = row + diag
            base1 = rna[row]
            base2 = rna[col]
            if row != col:
                # Use bp_score instead of constant values
                matrix[row][col] = abs(bp_score(base1, base2))
    
    stems_potential = []
    mu = 0

    for row in range(0, len(matrix)):
        for col in range (row, len(matrix)):
            if row != col:
                if matrix[row][col] != 0:
                    temp_row = row
                    temp_col = col
                    stem = [row+1,col+1,0]
                    length_N = 0
                    length_H = 0
                    while (matrix[temp_row][temp_col] != 0) and (temp_row != temp_col):
                        length_N += 1
                        # Calculate stack energy if possible
                        if temp_row + 1 < len(rna) and temp_col - 1 >= 0:
                            stack_key = (rna[temp_row], rna[temp_col], 
                                       rna[temp_row + 1], rna[temp_col - 1])
                            if stack_key in STACK_ENERGY:
                                length_H += abs(STACK_ENERGY[stack_key])
                            else:
                                length_H += matrix[temp_row][temp_col]
                        else:
                            length_H += matrix[temp_row][temp_col]
                        temp_row += 1
                        temp_col -= 1
                        if length_N >= 3:
                            stem[2] = int(length_H)
                            stems_potential.append(stem.copy())
                    if length_H > mu:
                        mu = length_H
    
    return [stems_potential, mu, rna, len(rna)]


def potential_pseudoknots(stems_potential, pkp):
    """
    Generate list of potential stem pairs that form pseudoknots
    
    Args:
        stems_potential (list): List of potential stems
        pkp (float): Pseudoknot penalty value
        
    Returns:
        list: List of potential pseudoknots
    """
    pseudoknots_potential = []
    pseudoknot_penalty = pkp

    for i in range(len(stems_potential)):
        for j in range(i + 1, len(stems_potential)):
            
            stem1 = stems_potential[i]
            stem2 = stems_potential[j]
    
            i_a = stem1[0]
            j_a = stem1[1]
            i_b = stem2[0]
            j_b = stem2[1]
    
            pseudoknot = [i,j,1]
    
            if (i_a < i_b and i_b < j_a and j_a < j_b) or (i_b < i_a and i_a < j_b and j_b < j_a):
                pseudoknot[2] = pseudoknot_penalty
    
            pseudoknots_potential.append(pseudoknot)
            
    return pseudoknots_potential


def potential_overlaps(stems_potential):
    """
    Generate list of stem pairs that overlap
    
    Args:
        stems_potential (list): List of potential stems
        
    Returns:
        list: List of potential overlaps
    """
    overlaps_potential = []
    overlap_penalty = 1e6

    for i in range(len(stems_potential)):
        for j in range(i+1, len(stems_potential)):
    
            stem1 = stems_potential[i]
            stem2 = stems_potential[j]
    
            overlap = [i, j, 0]
    
            stem1_cspan1 = set(range(stem1[1]-int(stem1[2])+1, stem1[1]+1))
            stem2_cspan1 = set(range(stem2[1]-int(stem2[2])+1, stem2[1]+1))
            
            stem1_cspan2 = set(range(stem1[0], stem1[0]+int(stem1[2])))
            stem2_cspan2 = set(range(stem2[0], stem2[0]+int(stem2[2])))
    
            if (len(stem1_cspan1 & stem2_cspan1) != 0) or (len(stem1_cspan2 & stem2_cspan2) != 0)  or (len(stem1_cspan1 & stem2_cspan2) != 0) or (len(stem1_cspan2 & stem2_cspan1) != 0):
                overlap[2] = overlap_penalty
        
            overlaps_potential.append(overlap)
            
    return overlaps_potential


def model(stems_potential, pseudoknots_potential, overlaps_potential, mu):
    """
    Generate the Hamiltonian of a given RNA structure from potential stems, overlaps, and pseudoknots
    
    Args:
        stems_potential (list): List of potential stems
        pseudoknots_potential (list): List of potential pseudoknots
        overlaps_potential (list): List of potential overlaps
        mu (float): Maximum stem energy
        
    Returns:
        tuple: Linear (L) and quadratic (Q) terms of the Hamiltonian
    """
    L = {}
    Q = {}
    cl = 1
    cb = 1
    k = 0

    for i in range(0, len(stems_potential)):
        # Add loop penalty to linear terms
        stem = stems_potential[i]
        loop = stem[1] - stem[0] - 1
        loop_penalty = LOOP_PENALTY.get(loop, 3.0) if 3 <= loop <= 30 else 0
        
        L[str(i)] = (cl*((stems_potential[i][2]**2)-2*mu*stems_potential[i][2]+mu**2)
                    -cb*(stems_potential[i][2]**2) + loop_penalty)
        
        for j in range(i+1, len(stems_potential)):
            Q[(str(i), str(j))] = (-2*cb*stems_potential[i][2]*stems_potential[j][2]
                                  *pseudoknots_potential[k][2]+overlaps_potential[k][2])
            k += 1
    
    return L, Q


