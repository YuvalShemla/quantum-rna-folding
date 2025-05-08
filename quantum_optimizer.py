"""
Quantum Optimizer for QUBO Problems using Qiskit
────────────────────────────────────────────────
Adds verbose debug instrumentation to help trace:

1. QUBO → Ising algebra (diagonal terms, constant shift, h_i, J_ij)
2. Agreement with QuadraticProgram.to_ising()
3. What SamplingVQE actually returns (parameters vs. bitstrings)
"""

DEBUG = True           # ← flip to False to mute all debug prints

from qiskit import QuantumCircuit          # kept for parity with your imports
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer   # still unused, kept
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from itertools import product
import numpy as np


# ────────────────────────────────────────────────────────────────────────────────
# 1. Build QUBO
# ────────────────────────────────────────────────────────────────────────────────
def build_quadratic_program(L, Q):
    """Build a QuadraticProgram from linear and quadratic terms.
    
    Args:
        L (dict): Linear terms with variable names as keys (int or str)
        Q (dict): Quadratic terms with tuples of variable names as keys (int or str)
    """
    qp = QuadraticProgram()
    
    # Get all unique variable names and convert to strings
    var_names = set()
    for k in L.keys():
        var_names.add(f"x_{k}" if isinstance(k, int) else k)
    for (i, j) in Q.keys():
        var_names.add(f"x_{i}" if isinstance(i, int) else i)
        var_names.add(f"x_{j}" if isinstance(j, int) else j)
    
    # Create binary variables
    for var_name in sorted(var_names):  # Sort for deterministic order
        qp.binary_var(name=var_name)
    
    # Convert L and Q to use string names
    L_str = {f"x_{k}" if isinstance(k, int) else k: v for k, v in L.items()}
    Q_str = {(f"x_{i}" if isinstance(i, int) else i, 
              f"x_{j}" if isinstance(j, int) else j): v 
             for (i, j), v in Q.items()}
    
    # Set objective
    qp.minimize(
        linear=L_str,
        quadratic=Q_str
    )
    return qp


# ────────────────────────────────────────────────────────────────────────────────
# 2. Manual QUBO → Ising (with diagnostics)
# ────────────────────────────────────────────────────────────────────────────────
def qubo_to_ising_manual(qp):
    n = qp.get_num_vars()
    linear    = np.zeros(n)
    quadratic = np.zeros((n, n))

    # Collect linear
    for i, v in qp.objective.linear.to_dict().items():
        linear[i] = v
    # Collect quadratic
    for (i, j), v in qp.objective.quadratic.to_dict().items():
        quadratic[i, j] = quadratic[j, i] = v

    if DEBUG:
        print("\n[DBG]  Raw L_i      :", linear)
        print("[DBG]  Raw Q_ij mat.:")
        print(quadratic)

    # Diagonal Q_ii are conventionally folded into L_i
    linear += np.diag(quadratic)
    np.fill_diagonal(quadratic, 0.0)

    # Ising algebra (corrected sign convention)
    J = quadratic / 4.0
    h = -linear / 2.0 - np.sum(quadratic, axis=1) / 4.0
    constant = -np.sum(linear)/2.0 + np.sum(np.triu(quadratic, 1))/4.0

    if DEBUG:
        print("\n[DBG]  h_i vector   :", h)
        print("[DBG]  non-zero J_ij :", {(i, j): J[i, j]
              for i in range(n) for j in range(i+1, n) if abs(J[i, j]) > 1e-12})
        print("[DBG]  Constant shift:", constant)

    # Build SparsePauliOp
    paulis, coeffs = [], []
    for i, h_i in enumerate(h):
        if abs(h_i) > 1e-12:
            paulis.append("I"*i + "Z" + "I"*(n-i-1))
            coeffs.append(float(h_i))
    for i in range(n):
        for j in range(i+1, n):
            if abs(J[i, j]) > 1e-12:
                z = ["I"]*n
                z[i] = z[j] = "Z"
                paulis.append("".join(z))
                coeffs.append(float(J[i, j]))

    return SparsePauliOp(paulis, coeffs) if paulis else SparsePauliOp("I"*n, [0.0]), constant


# ────────────────────────────────────────────────────────────────────────────────
# 3. Solve with SamplingVQE
# ────────────────────────────────────────────────────────────────────────────────
class Result:
    """Mimic your earlier return type."""
    def __init__(self, x, fval, status="SUCCESS"):
        self.x = x
        self.fval = fval
        self.status = status


def solve_with_vqe(qp, max_evals=1000):
    # Manual Ising build
    operator, offset = qubo_to_ising_manual(qp)

    # Qiskit's reference converter (for comparison)
    ref_op, ref_offset = qp.to_ising()
    if DEBUG:
        diff = (operator - ref_op).simplify()
        print("\n[DBG]  Manual vs. qp.to_ising():")
        print("       max|Δcoeff| =", np.max(np.abs(diff.coeffs)))
        print("       constant Δ  =", offset - ref_offset)

    # Ansatz & VQE
    ansatz = RealAmplitudes(qp.get_num_vars(), reps=2)
    optimizer = COBYLA(maxiter=max_evals)
    sampler = Sampler(options={"shots": 2048})
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)

    if DEBUG:
        print("\n[DBG]  Launching SamplingVQE …")

    eig = vqe.compute_minimum_eigenvalue(operator)

    if DEBUG:
        print("[DBG]  optimal_point (θ parameters) :", eig.optimal_point)
        print("[DBG]  optimal_value :", eig.optimal_value)

    # Get the most probable bitstring from the eigenstate
    probs = eig.eigenstate.binary_probabilities()
    best_bitstr = max(probs, key=probs.get)          # e.g. '010'
    bits = np.array([int(b) for b in best_bitstr[::-1]])  # Qiskit is little-endian
    spins = 2*bits - 1                                   # {0,1} → {-1,+1}
    fval = eig.eigenvalue.real + offset

    if DEBUG:
        print("[DBG]  Best Ising energy :", eig.optimal_value)
        print("[DBG]  Offset            :", offset)
        print("[DBG]  Final QUBO f(x)   :", fval)

    return Result(bits.astype(int), float(fval))


# ────────────────────────────────────────────────────────────────────────────────
# 4. Utility
# ────────────────────────────────────────────────────────────────────────────────
def analyze_result(res):
    print("\n──────── Optimisation summary ────────")
    print("Status          :", res.status)
    print("Objective value :", res.fval)
    print("Bitstring (x)   :", res.x)
    print("──────────────────────────────────────")


# ────────────────────────────────────────────────────────────────────────────────
# 5. Demo
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Toy QUBO
    L = {0: 1.0, 1: -2.0, 2: 0.5}
    Q = {(0, 1): 0.8, (1, 2): -1.2}

    qp = build_quadratic_program(L, Q)
    print("QUBO LP representation:\n", qp.export_as_lp_string())

    result = solve_with_vqe(qp, max_evals=300)
    analyze_result(result)
