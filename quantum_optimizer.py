"""
Quantum Optimizer for QUBO Problems using Qiskit
────────────────────────────────────────────────
Uses Qiskit's built-in functionality for QUBO to Ising conversion
and VQE optimization.

Important Notes:
1. Offset Handling: The energy returned is E_min + offset. We need to subtract the offset
   to get the actual QUBO objective value.
2. Solution Decoding: SamplingVQE returns a measurement distribution. We take the bitstring
   with highest probability as our solution.
3. Classical Budget: COBYLA optimizer may need hundreds-thousands of evaluations for
   non-trivial QUBOs. Adjust max_evals accordingly.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit.primitives import Sampler
import numpy as np

# Configuration
ANSATZ_TYPE = "RealAmplitudes"  # Options: "RealAmplitudes", "TwoLocal", "EfficientSU2"
MAX_EVALS = 10000
SHOTS = 2048

def get_ansatz(num_qubits, ansatz_type=ANSATZ_TYPE):
    """Get the specified ansatz circuit.
    
    Args:
        num_qubits (int): Number of qubits in the circuit
        ansatz_type (str): Type of ansatz to use
        
    Returns:
        QuantumCircuit: The parameterized quantum circuit
    """
    if ansatz_type == "RealAmplitudes":
        return RealAmplitudes(num_qubits, reps=2)
    
    elif ansatz_type == "TwoLocal":
        return TwoLocal(num_qubits, 
                       rotation_blocks=['ry', 'rz'],
                       entanglement_blocks='cx',
                       entanglement='linear',
                       reps=2)
    
    elif ansatz_type == "EfficientSU2":
        return EfficientSU2(num_qubits, 
                          su2_gates=['ry', 'rz'],
                          entanglement='linear',
                          reps=2)
    
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

def build_quadratic_program(L, Q):
    """
    Build a QuadraticProgram from linear (L) and quadratic (Q) dictionaries.
    Keys in L and Q are integers → they become variable indices v0, v1, …
    """
    qp = QuadraticProgram()

    # ---------- 1) collect and sort unique indices ----------
    idx_set = set(L.keys()) | {i for (i, _) in Q.keys()} | {j for (_, j) in Q.keys()}
    idx_list = sorted(idx_set)          # deterministic order 0,1,2,…

    # ---------- 2) create binary variables v0, v1, … ----------
    for idx in idx_list:
        qp.binary_var(name=f"v{idx}")

    # ---------- 3) build objective using the same names ----------
    lin = {f"v{idx}": coeff for idx, coeff in L.items()}
    quad = {(f"v{i}", f"v{j}"): coeff for (i, j), coeff in Q.items()}

    qp.minimize(linear=lin, quadratic=quad)
    return qp


class Result:
    """Container for optimization results."""
    def __init__(self, x, fval, status="SUCCESS", probabilities=None):
        self.x = x                    # Binary solution vector
        self.fval = fval             # Objective value (QUBO energy)
        self.status = status         # Optimization status
        self.probabilities = probabilities  # Measurement probabilities


def solve_with_vqe(qp, max_evals=MAX_EVALS, ansatz_type=ANSATZ_TYPE):
    """Solve QUBO using VQE.
    
    Args:
        qp (QuadraticProgram): The QUBO problem to solve
        max_evals (int): Maximum number of function evaluations for the optimizer
        ansatz_type (str): Type of ansatz to use
        
    Returns:
        Result: Container with solution, objective value, and measurement probabilities
    """
    # Convert QUBO to Ising using Qiskit's built-in functionality
    operator, offset = qp.to_ising()

    # Setup VQE components
    num_qubits = qp.get_num_vars()
    ansatz = get_ansatz(num_qubits, ansatz_type)
    optimizer = COBYLA(maxiter=max_evals)
    sampler = Sampler(options={"shots": SHOTS})
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)

    # Solve for minimum eigenvalue
    print(f"Running VQE with {ansatz_type} ansatz and COBYLA optimizer...")
    eig = vqe.compute_minimum_eigenvalue(operator)

    # Get measurement probabilities for all bitstrings
    probs = eig.eigenstate.binary_probabilities()
    
    # Find the bitstring with highest probability (lowest energy)
    best_bitstr = max(probs, key=probs.get)
    
    # Convert bitstring to binary array (keeping consistent endianness)
    bits = np.array([int(b) for b in best_bitstr])
    
    # Calculate final QUBO objective value
    fval = eig.eigenvalue.real + offset

    return Result(
        x=bits.astype(int),
        fval=float(fval),
        probabilities=probs
    )


def analyze_result(res):
    """Print optimization results in a readable format.
    
    Args:
        res (Result): The optimization result to analyze
    """
    print("\n──────── Optimisation summary ────────")
    print("Status          :", res.status)
    print("Objective value :", res.fval)
    print("Bitstring (x)   :", res.x)
    
    if res.probabilities:
        # Show top 3 most probable solutions
        top_probs = sorted(res.probabilities.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:3]
        print("\nTop 3 solutions by probability:")
        for bitstr, prob in top_probs:
            # Convert probability bitstring to same format as result
            bits = np.array([int(b) for b in bitstr])
            print(f"  {bitstr}: {prob:.4f} -> {bits}")
    
    print("──────────────────────────────────────")


if __name__ == "__main__":
    # Toy QUBO example with 8 stems
    # Linear terms represent individual stem energies
    L = {
        0: -2.5,  # Very strong GC stem
        1: -2.0,  # Strong GC stem
        2: -1.8,  # Strong GC-AU mixed stem
        3: -1.5,  # Medium GC stem
        4: -1.2,  # Medium AU-GC stem
        5: -1.0,  # Weak AU stem
        6: -0.8,  # Very weak AU stem
        7: -0.5   # Weakest AU stem
    }
    
    # Quadratic terms represent stem interactions (penalties for incompatible stems)
    Q = {
        (0, 1): 2.5,   # Strong penalty - overlapping GC stems
        (1, 2): 100,   # Strong penalty - overlapping mixed stems
        (2, 3): 1.8,   # Medium penalty - partially overlapping
        (3, 4): 1.5,   # Medium penalty - competing stems
        (4, 5): 100,   # Weak penalty - distant stems
        (5, 6): 1.0,   # Weak penalty - AU stem interaction
        (0, 7): 2.2,   # Strong penalty - pseudoknot formation
        (2, 6): 100    # Medium penalty - structural conflict
    }

    print("=== Running Enhanced Toy Model (8 Stems) ===")
    print("\nLinear Terms (Stem Energies):")
    for i, energy in L.items():
        print(f"Stem {i}: {energy:5.2f} - {'Very Strong' if energy <= -2.2 else 'Strong' if energy <= -1.8 else 'Medium' if energy <= -1.2 else 'Weak'}")
    
    print("\nQuadratic Terms (Stem Interactions):")
    for (i, j), penalty in Q.items():
        print(f"Stems {i}-{j}: {penalty:4.1f} - {'Strong penalty' if penalty >= 2.0 else 'Medium penalty' if penalty >= 1.5 else 'Weak penalty'}")

    qp = build_quadratic_program(L, Q)
    print("\nQUBO LP representation:\n", qp.export_as_lp_string())

    # Test standard VQE
    print("\n=== Testing Standard VQE ===")
    result_standard = solve_with_vqe(qp, max_evals=800)
    analyze_result(result_standard)
