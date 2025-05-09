"""
Quantum Optimizer for QUBO Problems using Qiskit
────────────────────────────────────────────────
Uses Qiskit's built-in functionality for QUBO to Ising conversion
and VQE optimization.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit.primitives import Sampler
import numpy as np

# Configuration
ANSATZ_TYPE = "TwoLocal"  # Options: "RealAmplitudes", "TwoLocal", "EfficientSU2"
MAX_EVALS = 1000
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
    eig = vqe.compute_minimum_eigenvalue(operator)

    # Get measurement probabilities for all bitstrings
    probs = eig.eigenstate.binary_probabilities()
    
    # Find the bitstring with highest probability (lowest energy)
    best_bitstr = max(probs, key=probs.get)
    
    # Convert bitstring to binary array (note: Qiskit uses little-endian)
    bits = np.array([int(b) for b in best_bitstr[::-1]])
    
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
            print(f"  {bitstr}: {prob:.4f}")
    
    print("──────────────────────────────────────")


if __name__ == "__main__":
    # Toy QUBO example
    L = {0: 1.0, 1: -2.0, 2: 0.5}
    Q = {(0, 1): 0.8, (1, 2): -1.2}

    qp = build_quadratic_program(L, Q)
    print("QUBO LP representation:\n", qp.export_as_lp_string())

    # For this small problem, 300 evaluations is sufficient
    # For larger problems, consider increasing max_evals
    result = solve_with_vqe(qp, max_evals=300)
    analyze_result(result)
