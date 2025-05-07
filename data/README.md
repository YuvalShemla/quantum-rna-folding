# RNA Structure Data Formats

## FASTA Format
The FASTA file contains the primary sequence of the RNA molecule.

### Format
```
>RF01072_D13438.1_6370-6402
AGUGUUUUUUUCCCUCCACUUAAAUCGAAGGGU
```

### Header Information
- `RF01072`: Rfam family ID
- `D13438.1`: GenBank accession number
- `6370-6402`: Sequence position in original genome

### Sequence
- Contains the linear sequence of nucleotides (A, U, G, C)
- Length: 33 nucleotides
- Represents the primary structure of the RNA

## CT (Connectivity Table) Format
The CT file contains detailed information about the RNA's secondary structure.

### Format
Each line contains 6 columns:
```
1    A    0    2    0    1
2    G    1    3    19   2
3    U    2    4    18   3
...
```

### Column Meanings
1. **Nucleotide Number**: Position in sequence (1-based)
2. **Base**: Nucleotide type (A, U, G, C)
3. **Previous**: Index of previous nucleotide
4. **Next**: Index of next nucleotide
5. **Pair**: Index of paired nucleotide (0 if unpaired)
6. **Historical**: Historical numbering (usually same as column 1)

### Example
```
2    G    1    3    19   2
```
- Position 2 in sequence
- Contains Guanine (G)
- Connected to nucleotide 1 before it
- Connected to nucleotide 3 after it
- Pairs with nucleotide 19
- Historical number is 2

## Usage in RNA Structure Prediction
- FASTA provides the primary sequence
- CT file provides the known secondary structure
- Together they help:
  - Identify known stems (base pairs)
  - Understand folding patterns
  - Validate structure predictions
  - Train quantum algorithms 