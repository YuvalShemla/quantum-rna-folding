# RNA Hamiltonian VQE Template using Qiskit
# --------------------------------------------------
# Full pipeline:
#  0) Preprocess an RNA sequence to QUBO coefficients and sets
#  1) Build parameterized ansatz circuit based on number of quartets
#  2) Measure necessary Pauli-Z observables from those sets
#  3) Compute energy expectation
#  4) Optimize variational parameters via brute force (or swap in classical optimizer)
#  5) (Optional) Visualize results and inspect sets

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# =============================================================================
# 0) Preprocessing: From RNA sequence to QUBO coefficients and sets
# =============================================================================
def preprocess_sequence(sequence, r, p, t):
    """
    Input:
      - sequence: RNA string (e.g. 'GCGUAUCG')
      - r    : reward weight for stacking quartets
      - p    : penalty weight for UA-end motifs
      - t    : penalty weight for crossing quartets

    Outputs a dict containing:
      - 'linear': {i: a_i} coefficients for Z_i
      - 'quadratic': {(i,j): b_ij} coefficients for Z_i Z_j
      - '_Q': list of quartets Q (each as (start,end) indices)
      - '_QS': set of stacking pairs
      - '_QUA': set of UA-end penalty pairs
      - '_QC': set of crossing constraint pairs
    """
    Q = []
    # Identify candidate quartets (stack length 2 motif)
    for i in range(len(sequence)-3):
        j = i + 3
        pair1 = (sequence[i], sequence[j])
        pair2 = (sequence[i+1], sequence[j-1])
        valid = [('G','C'),('C','G'),('A','U'),('U','A')]
        if pair1 in valid and pair2 in valid:
            Q.append((i, j))

    # Mock free energies for each quartet; replace with real data lookup
    e_qi = {idx: -1.0 + 0.1*idx for idx in range(len(Q))}

    QS, QUA, QC = set(), set(), set()
    for i, qi in enumerate(Q):
        for j, qj in enumerate(Q):
            if i >= j: continue
            # stacking if consecutive quartets
            if qi[1] == qj[0]:
                QS.add((i, j))
            # UA-end penalty if qi ends in U,A
            if sequence[qi[0]]=='U' and sequence[qi[1]]=='A':
                QUA.add((i, j))
            # crossing if intervals interleave
            if (qi[0] < qj[0] < qi[1] < qj[1]) or (qj[0] < qi[0] < qj[1] < qi[1]):
                QC.add((i, j))

    # Build QUBO coefficients
    linear, quadratic = {}, {}
    # Pre-count QUA contributions for linear term
    qua_count = {i:0 for i in range(len(Q))}
    for (i,j) in QUA:
        qua_count[i] += 1  # from term p * qi(1 - qj) -> linear in qi

    # Linear: e_qi + p * count
    for i in range(len(Q)):
        linear[i] = e_qi[i] + p * qua_count[i]

    # Quadratic: r for QS, -p for QUA, +t for QC
    for i in range(len(Q)):
        for j in range(i+1, len(Q)):
            b = 0
            if (i,j) in QS:  b += r
            if (i,j) in QUA: b -= p
            if (i,j) in QC:  b += t
            if b != 0:
                quadratic[(i,j)] = b

    return {
        'linear': linear,
        'quadratic': quadratic,
        '_Q': Q,
        '_QS': QS,
        '_QUA': QUA,
        '_QC': QC
    }

# =============================================================================
# 1) Build the RNA Hamiltonian from a sequence
# =============================================================================
sequence = 'GCGUAUCG'
r = 0.4   # stacking reward
p = 0.3   # UA-end penalty
t = 1.0   # crossing penalty
ham = preprocess_sequence(sequence, r, p, t)
Q, QS, QUA, QC = ham['_Q'], ham['_QS'], ham['_QUA'], ham['_QC']
num_qubits = len(Q)
print(f"Detected {num_qubits} quartets: {Q}")
print(f"Stacking pairs (QS): {QS}")
print(f"UA-end penalties (QUA): {QUA}")
print(f"Crossing constraints (QC): {QC}")

# =============================================================================
# 2) Parameterized Ansat z Circuit based on num_qubits
#    Here we use a simple single-qubit ry rotations + linear entangling chain
# =============================================================================
def build_ansatz(params):
    qreg = QuantumRegister(num_qubits)
    creg = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qreg, creg)

    # Single-qubit rotations
    for i in range(num_qubits):
        qc.ry(params[i], qreg[i])
    # Entangle in a chain
    for i in range(num_qubits-1):
        qc.cx(qreg[i], qreg[i+1])

    return qc

# =============================================================================
# 3) Measurement: compute <Z_i> and <Z_i Z_j> as required
# =============================================================================
def measure_Z_terms(qc, qubits, shots=1024):
    # copy to avoid side-effects
    qc_meas = qc.copy()
    qc_meas.measure_all()
    backend = AerSimulator()
    counts = backend.run(qc_meas, shots=shots).result().get_counts()

    exp_val = 0
    for bitstr, cnt in counts.items():
        # Qiskit bitstr is reversed order
        prod = 1
        for q in qubits:
            if bitstr[::-1][q] == '1': prod *= -1
        exp_val += prod * cnt/shots
    return exp_val

# =============================================================================
# 4) Expectation Value of the RNA Hamiltonian
# =============================================================================
def expectation(params, ham, shots=1024):
    qc = build_ansatz(params)
    energy = 0
    # linear terms
    for i, coeff in ham['linear'].items():
        energy += coeff * measure_Z_terms(qc, [i], shots)
    # quadratic terms
    for (i,j), coeff in ham['quadratic'].items():
        energy += coeff * measure_Z_terms(qc, [i,j], shots)
    return energy

# =============================================================================
# 5) Brute-force (grid) optimization over parameters
#    For num_qubits, use a grid of size steps per parameter -> computationally
#    expensive for >3 qubits. Swap to classical optimizer for larger.
# =============================================================================
def brute_force_opt(ham, steps=10, shots=1024):
    # grid from 0 to 2pi for each param
    grid = np.linspace(0, 2*np.pi, steps)
    best = {'energy':np.inf, 'params':None}

    # recursive loops over each qubit's angle
    def recurse(idx, current_params):
        if idx == num_qubits:
            E = expectation(current_params, ham, shots)
            if E < best['energy']:
                best.update({'energy':E, 'params':current_params.copy()})
            return
        for angle in grid:
            current_params[idx] = angle
            recurse(idx+1, current_params)

    recurse(0, [0]*num_qubits)
    print(f"Optimal energy {best['energy']:.4f} at params {best['params']}")
    return best

# =============================================================================
# 6) Run the VQE simulation
# =============================================================================
if __name__ == '__main__':
    best_result = brute_force_opt(ham, steps=20, shots=2048)

# =============================================================================
# 7) Next Steps for Experimental Relevance
#    - Swap brute_force_opt for gradient-based or heuristic optimizers
#    - Incorporate noise mitigation: readout calibration, zero-noise extrap.
#    - Transpile circuits for target hardware topology, include layout.
#    - Benchmark against classical solvers on same instances.
#    - Increase model complexity: include multi-body terms or kinetic contributions.
# =============================================================================
