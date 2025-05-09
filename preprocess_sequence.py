"""
Preprocess the RNA sequence to get the QUBO coefficients and sets
Here we use the paper to understand how to preprocess each term
"""
import numpy as np

# Constants for RNA structure prediction
DEFAULT_STACK_ENERGY = {
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

DEFAULT_LOOP_PENALTY = {3: 5.4, 4: 4.1, 5: 3.4, 6: 3.0}

def default_bp_score(b1, b2):
    """Default base pair scoring function."""
    if (b1,b2) in [('G','C'),('C','G')]: return -3.4
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
    
    rna = fasta_lines[1].strip().upper()  # Strip newlines and convert to uppercase
    
    stems_actual = []
    sip = False                       # stem in progress?
    sl = 0                           # stem length
    last_line = [0, 0, 0, 0, 0, 0]    # initiate last line

    for i in range(0, len(lines)):
        line = lines[i].strip().split()
        if (int(line[4]) != 0 and sip == False):
            sip = True
            temp = [int(line[0]), int(line[4])]
            # Use proper set membership test
            if (rna[i] in ('G', 'C') and rna[int(line[4])-1] in ('G', 'C')):
                sl = 3
            elif (rna[i] in ('A', 'U', 'G') and rna[int(line[4])-1] in ('A', 'U', 'G')):
                sl = 2
        if (int(line[4]) != 0 and sip == True and (int(last_line[4])-int(line[4]) == 1)):
            if (rna[i] in ('G', 'C') and rna[int(line[4])-1] in ('G', 'C')):
                sl += 3
            elif (rna[i] in ('A', 'U', 'G') and rna[int(line[4])-1] in ('A', 'U', 'G')):
                sl += 2
        if (int(line[4]) == 0 and sip == True):
            sip = False
            temp.append(sl)
            if temp[1] > temp[0]:
                stems_actual.append(temp)
            sl = 0
        if ((int(last_line[4])-int(line[4]) != 1) and int(last_line[4]) != 0 and sip == True):
            temp.append(sl)
            if temp[1] > temp[0]:
                stems_actual.append(temp)
            temp = [int(line[0]), int(line[4])]
            sl = 0  # Reset sl before potentially setting new value
            if (rna[i] in ('G', 'C') and rna[int(line[4])-1] in ('G', 'C')):
                sl = 3
            elif (rna[i] in ('A', 'U', 'G') and rna[int(line[4])-1] in ('A', 'U', 'G')):
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
    
    rna = lines[1].strip().upper()  # Strip newlines and convert to uppercase
    
    matrix = np.zeros((len(rna),len(rna)))
    for diag in range(0, len(matrix)):
        for row in range(0, len(matrix)-diag):
            col = row + diag
            base1 = rna[row]
            base2 = rna[col]
            if row != col:
                matrix[row][col] = bp_score(base1, base2)  # No abs() here
    
    stems_potential = []
    mu = 0

    for row in range(0, len(matrix)):
        for col in range(row, len(matrix)):
            if row != col and matrix[row][col] != 0:
                temp_row = row
                temp_col = col
                stem = [row+1, col+1, 0]
                length_N = 0
                length_H = 0
                while (matrix[temp_row][temp_col] != 0) and (temp_row != temp_col):
                    length_N += 1
                    if temp_row + 1 < len(rna) and temp_col - 1 >= 0:
                        stack_key = (rna[temp_row], rna[temp_col], 
                                   rna[temp_row + 1], rna[temp_col - 1])
                        if stack_key in DEFAULT_STACK_ENERGY:
                            length_H += abs(DEFAULT_STACK_ENERGY[stack_key])
                        else:
                            length_H += abs(matrix[temp_row][temp_col])
                    else:
                        length_H += abs(matrix[temp_row][temp_col])
                    temp_row += 1
                    temp_col -= 1
                    
                    # Only update stem and mu if we have a valid stem length
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
    overlap_penalty = 10000 # try different values

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
        loop_penalty = DEFAULT_LOOP_PENALTY.get(loop, 3.0) if 3 <= loop <= 30 else 0
        
        L[str(i)] = (cl*((stems_potential[i][2]**2)-2*mu*stems_potential[i][2]+mu**2)
                    -cb*(stems_potential[i][2]**2) + loop_penalty)
        
        for j in range(i+1, len(stems_potential)):
            Q[(str(i), str(j))] = (-2*cb*stems_potential[i][2]*stems_potential[j][2]
                                  *pseudoknots_potential[k][2]+overlaps_potential[k][2])
            k += 1
    
    return L, Q

    
# function to evaluate the energy of the known structure under the model Hamiltonian:
def energy(stems_actual, pkp):
    cl = 1
    cb = 1
    k = 0
    
    pseudoknots_actual = potential_pseudoknots(stems_actual, pkp)
    cost = 0
    mu = max(list(map(list, zip(*stems_actual)))[2])
    
    for i in range(0, len(stems_actual)):
        cost += cl*((stems_actual[i][2]**2)-2*mu*stems_actual[i][2]+mu**2)-cb*(stems_actual[i][2]**2)
        for j in range(i+1, len(stems_actual)):
            cost -= 2*cb*stems_actual[i][2]*stems_actual[j][2]*pseudoknots_actual[k][2]
            k += 1
    
    return cost


# function to compare actual and predicted structure based on comparison of base-pairs:
def evaluation_1(stems_actual, stems_potential):
    bp_actual = []
    bp_predicted = []

    for i in range(0, len(stems_actual)):
        for j in range(0, stems_actual[i][2]):
            bp_actual.append((stems_actual[i][0]+j, stems_actual[i][1]-j))
        
    for i in range(0, len(stems_potential)):
        for j in range(0, stems_potential[i][2]):
            bp_predicted.append((stems_potential[i][0]+j, stems_potential[i][1]-j))
            
    C = 0    # number of correctly identified base pairs
    M = 0    # number of the predicted base pairs missing from the known structure
    I = 0    # number of non-predicted base pairs present in the known structure

    for i in range(0, len(bp_predicted)):
        if bp_predicted[i] in bp_actual:
            C += 1
        else:
            M += 1

    for i in range(0, len(bp_actual)):
        if bp_actual[i] not in bp_predicted:
            I += 1
            
    sensitivity = C/(C+M)
    specificity = C/(C+I)
    
    return [sensitivity, specificity]


# function to compare actual and predicted structure based on comparison of bases involved in pairing:
def evaluation_2(stems_actual, stems_predicted):
    
    b_actual = []
    b_predicted = []

    for i in range(0, len(stems_actual)):
        for j in range(0, stems_actual[i][2]):
            b_actual.append(stems_actual[i][0]+j)
            b_actual.append(stems_actual[i][1]-j)
        
    for i in range(0, len(stems_predicted)):
        for j in range(0, stems_predicted[i][2]):
            b_predicted.append(stems_predicted[i][0]+j)
            b_predicted.append(stems_predicted[i][1]-j)
            
    C = 0    # number of correctly identified bases that are paired
    M = 0    # number of the predicted paired bases missing from the known structure
    I = 0    # number of non-predicted paired bases present in the known structure

    for i in range(0, len(b_predicted)):
        if b_predicted[i] in b_actual:
            C += 1
        else:
            M += 1

    for i in range(0, len(b_actual)):
        if b_actual[i] not in b_predicted:
            I += 1
            
    sensitivity = C/(C+M)
    specificity = C/(C+I)
    
    return [sensitivity, specificity]
    

def stems_overlap(stem1, stem2):
    """Check if two stems share any positions."""
    stem1_positions = set([pos for pair in stem1[3] for pos in pair])
    stem2_positions = set([pos for pair in stem2[3] for pos in pair])
    return bool(stem1_positions.intersection(stem2_positions))

def forms_pseudoknot(stem1, stem2):
    """Check if two stems form a pseudoknot."""
    i_a, j_a = stem1[0], stem1[1]
    i_b, j_b = stem2[0], stem2[1]
    return (i_a < i_b and i_b < j_a and j_a < j_b) or (i_b < i_a and i_a < j_b and j_b < j_a)

def are_stems_compatible(stem1, stem2):
    """Check if two stems can coexist (no overlap, no pseudoknot)."""
    return not (stems_overlap(stem1, stem2) or forms_pseudoknot(stem1, stem2))

def process_rna_sequence(sequence, 
                        pseudoknot_penalty=2.0,
                        overlap_penalty=1000,
                        min_stem_length=3,
                        min_loop_size=3,
                        stack_energy=None,
                        loop_penalty=None,
                        bp_score_func=None,
                        cl=1,  # coefficient for linear terms
                        cb=1,  # coefficient for binary terms
                        scale_penalties=True):
    """
    Process an RNA sequence and return QUBO terms for optimization.
    """
    # Use default values if not provided
    stack_energy = stack_energy if stack_energy is not None else DEFAULT_STACK_ENERGY
    loop_penalty = loop_penalty if loop_penalty is not None else DEFAULT_LOOP_PENALTY
    bp_score_func = bp_score_func if bp_score_func is not None else default_bp_score
    
    # Convert sequence to uppercase
    sequence = sequence.upper()
    
    # Valid RNA base pairs
    valid_pairs = [('G','C'), ('C','G'), ('A','U'), ('U','A'), ('G','U'), ('U','G')]
    
    # Find potential stems
    stems_potential = []
    mu = 0
    
    # Scan for potential stems
    for i in range(len(sequence) - min_stem_length * 2 - min_loop_size):
        for j in range(i + min_stem_length * 2 + min_loop_size, len(sequence)):
            # Check if this pair can form a valid base pair
            if (sequence[i], sequence[j]) in valid_pairs:
                # Try to extend this pair into a stem
                stem_length = 0
                stem_energy = 0
                base_pairs = []  # Track base pairs for validation
                curr_start = i
                curr_end = j
                
                # Extend the stem as long as possible
                while (curr_start < curr_end - min_loop_size and  # Ensure minimum loop size
                       (sequence[curr_start], sequence[curr_end]) in valid_pairs):
                    # Add base pair
                    base_pairs.append((curr_start + 1, curr_end + 1))  # Store 1-based coordinates
                    
                    # Add base pair energy
                    pair_energy = abs(bp_score_func(sequence[curr_start], sequence[curr_end]))
                    stem_energy += pair_energy
                    
                    # Add stacking energy if possible and if we'll have another pair
                    if (curr_start + 1 < curr_end - 1 and
                        (sequence[curr_start + 1], sequence[curr_end - 1]) in valid_pairs):
                        stack_key = (sequence[curr_start], sequence[curr_end],
                                   sequence[curr_start + 1], sequence[curr_end - 1])
                        if stack_key in stack_energy:
                            stem_energy += abs(stack_energy[stack_key])
                    
                    stem_length += 1
                    curr_start += 1
                    curr_end -= 1
                
                # Only add stems that meet minimum length requirement and have valid loop size
                if stem_length >= min_stem_length:
                    loop_size = j - i + 1 - 2 * stem_length
                    if loop_size >= min_loop_size:
                        stem = [i + 1, j + 1, int(stem_energy), base_pairs]  # Include base pairs for validation
                        stems_potential.append(stem)
                        if stem_energy > mu:
                            mu = stem_energy
    
    # Build penalty dictionaries
    pseudoknot_dict = {}
    overlap_dict = {}
    
    # Calculate pseudoknot and overlap penalties
    for i in range(len(stems_potential)):
        for j in range(i + 1, len(stems_potential)):
            stem1 = stems_potential[i]
            stem2 = stems_potential[j]
            
            # Check for overlaps using actual base pairs
            stem1_positions = set([pos for pair in stem1[3] for pos in pair])
            stem2_positions = set([pos for pair in stem2[3] for pos in pair])
            
            if stem1_positions.intersection(stem2_positions):
                overlap_dict[(i,j)] = overlap_penalty
            else:
                overlap_dict[(i,j)] = 0
            
            # Check for pseudoknots using actual base pairs
            i_a, j_a = stem1[0], stem1[1]
            i_b, j_b = stem2[0], stem2[1]
            
            if (i_a < i_b and i_b < j_a and j_a < j_b) or (i_b < i_a and i_a < j_b and j_b < j_a):
                pseudoknot_dict[(i,j)] = pseudoknot_penalty
            else:
                pseudoknot_dict[(i,j)] = 0
    
    # Generate QUBO terms
    L = {}
    Q = {}
    
    # Make linear terms negative (reward for selecting stems)
    for i in range(len(stems_potential)):
        stem = stems_potential[i]
        H_i = stem[2]  # stem energy
        # Negative term encourages selection
        L[i] = -H_i
        
    # Calculate quadratic terms
    for i in range(len(stems_potential)):
        for j in range(i+1, len(stems_potential)):
            stem1 = stems_potential[i]
            stem2 = stems_potential[j]
            
            # Check if stems are compatible
            if are_stems_compatible(stem1, stem2):
                # Add bonus for compatible stems
                Q[(i,j)] = -0.1 * (stem1[2] + stem2[2])
            else:
                # Penalize conflicts
                if stems_overlap(stem1, stem2):
                    Q[(i,j)] = overlap_penalty
                elif forms_pseudoknot(stem1, stem2):
                    Q[(i,j)] = pseudoknot_penalty
                    
    return L, Q, stems_potential, sequence


def analyze_sequence(sequence):
    """
    Analyze an RNA sequence with detailed debugging output.
    """
    print("=== Starting RNA Sequence Analysis ===")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)} bases\n")

    # Process the sequence
    L, Q, stems, sequence = process_rna_sequence(sequence)

    # Detailed Stem Analysis
    print("Stem Analysis:")
    for i, stem in enumerate(stems):
        print(f"\nStem {i}:")
        print(f"  Start-End: {stem[0]}-{stem[1]}")
        print(f"  Energy: {stem[2]}")
        print("  Base pairs:")
        for start, end in stem[3]:  # Use stored base pairs
            print(f"    Position {start}: {sequence[start-1]} pairs with Position {end}: {sequence[end-1]}")
        
        # Calculate stem properties
        num_pairs = len(stem[3])
        span = stem[1] - stem[0] + 1  # Total bases spanned
        loop_size = span - 2 * num_pairs  # Bases not in pairs
        
        print(f"  Number of pairs: {num_pairs}")
        print(f"  Total span: {span} bases")
        print(f"  Loop size: {loop_size} bases")
        if loop_size < 3:
            print("  WARNING: Loop size < 3, may be unstable")

    # QUBO Terms Analysis
    print("\nQUBO Terms:")
    print("\nLinear Terms:")
    for stem_idx, value in L.items():
        stem = stems[stem_idx]
        loop_size = stem[1] - stem[0] + 1 - 2*len(stem[3])
        print(f"  Stem {stem_idx}: {value:.2f} (loop size: {loop_size})")
        
    print("\nQuadratic Terms:")
    for (i, j), value in Q.items():
        if value > 0:
            print(f"  Stems {i}-{j}: Overlap penalty = {value:.2f}")
        elif value < 0:
            print(f"  Stems {i}-{j}: Pseudoknot penalty = {value:.2f}")

    # Summary
    print("\nSummary:")
    print(f"Found {len(stems)} stems")
    if stems:
        print("Stem details:")
        for i, stem in enumerate(stems):
            num_pairs = len(stem[3])
            loop_size = stem[1] - stem[0] + 1 - 2*num_pairs
            print(f"  Stem {i}: {num_pairs} pairs, loop size {loop_size}")
    print(f"Maximum stem energy: {max(stem[2] for stem in stems) if stems else 'N/A'}")
    print(f"Number of overlapping pairs: {sum(1 for v in Q.values() if v > 0)}")
    print(f"Number of pseudoknots: {sum(1 for v in Q.values() if v < 0)}")

if __name__ == "__main__":
    # Test sequence
    sequence = "GCGUAAACGCG"
    print("\n=== Testing sequence ===")
    analyze_sequence(sequence)


