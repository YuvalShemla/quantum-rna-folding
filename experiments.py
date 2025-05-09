"""
RNA Folding Experiments
----------------------
This file contains experiments for analyzing RNA folding using classical and quantum approaches.
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from preprocess_sequence import process_rna_sequence
from classical_optimizer import solve_qubo_with_cpsat
from quantum_optimizer import build_quadratic_program, solve_with_vqe, get_ansatz
import signal
import time
from contextlib import contextmanager
import random
from mpl_toolkits.mplot3d import Axes3D

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_plot(plt, name):
    """Save plot with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{name}_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved as: {filename}")

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def generate_random_rna(length):
    """Generate a random RNA sequence of given length."""
    bases = ['G', 'C', 'A', 'U']
    # Ensure some GC content for stable structures
    sequence = ['G'] + [random.choice(bases) for _ in range(length-2)] + ['C']
    return ''.join(sequence)

def analyze_qubo_resources(max_length=100, step_size=1):
    """Analyze how QUBO resources scale with sequence length up to 100."""
    print("\n=== QUBO Resource Analysis ===")
    results = []
    
    for length in range(5, max_length + 1, step_size):
        print(f"\nAnalyzing sequence of length {length}")
        sequence = generate_random_rna(length)
        print(f"Sequence: {sequence}")
        
        L, Q, _, _ = process_rna_sequence(sequence)
        num_linear = len(L)
        num_quadratic = len(Q)
        
        results.append({
            'length': length,
            'sequence': sequence,
            'num_linear': num_linear,
            'num_quadratic': num_quadratic,
        })
        
        print(f"Linear terms: {num_linear}")
        print(f"Quadratic terms: {num_quadratic}")
    
    plot_qubo_scaling(results)
    return results

def generate_rna_dataset(examples_per_qubit=2):
    """Generate diverse RNA sequences and analyze their QUBO complexity."""
    sequences = []
    # Generate 10 examples for each length from 10 to 30
    for length in range(10, 31):
        for _ in range(10):  # Generate 10 examples per length
            sequence = generate_random_rna(length)
            L, Q, stems, _ = process_rna_sequence(sequence)
            # Skip sequences with 0 qubits
            if len(L) == 0:
                continue
            sequences.append({
                'sequence': sequence,
                'length': length,
                'num_linear': len(L),
                'num_quadratic': len(Q),
                'L': L,
                'Q': Q,
                'stems': stems
            })
    
    # Sort by number of linear terms (qubits)
    sequences.sort(key=lambda x: x['num_linear'])
    
    # Keep specified number of examples for each unique number of qubits
    unique_sequences = []
    current_qubits = None
    qubit_count = 0
    
    for seq in sequences:
        # Skip sequences with 0 qubits
        if seq['num_linear'] == 0:
            continue
            
        if seq['num_linear'] != current_qubits:
            current_qubits = seq['num_linear']
            qubit_count = 0
        
        if qubit_count < examples_per_qubit:  # Keep specified number of examples per number of qubits
            unique_sequences.append(seq)
            qubit_count += 1
    
    return unique_sequences

def analyze_vqe_runtime(timeout_minutes=5, examples_per_qubit=2):
    """Analyze VQE runtime based on QUBO complexity."""
    print("\n=== VQE Runtime Analysis ===")
    
    # Generate and sort sequences
    sequences = generate_rna_dataset(examples_per_qubit)
    results = []
    timeout_seconds = timeout_minutes * 60
    
    for seq_data in sequences:
        sequence = seq_data['sequence']
        print(f"\nAnalyzing sequence: {sequence}")
        print(f"Length: {seq_data['length']}")
        print(f"Linear terms: {seq_data['num_linear']}")
        
        # Build quadratic program
        qp = build_quadratic_program(seq_data['L'], seq_data['Q'])
        num_qubits = qp.get_num_vars()
        ansatz = get_ansatz(num_qubits)
        num_parameters = ansatz.num_parameters
        
        # Try to solve with timeout
        start_time = time.time()
        try:
            with timeout(timeout_seconds):
                result = solve_with_vqe(qp)
                runtime = time.time() - start_time
                status = "Completed"
                energy = result.fval
        except TimeoutException:
            runtime = timeout_seconds
            status = "Timeout"
            energy = None
        except Exception as e:
            runtime = time.time() - start_time
            status = f"Error: {str(e)}"
            energy = None
        
        results.append({
            'sequence': sequence,
            'length': seq_data['length'],
            'num_linear': seq_data['num_linear'],
            'num_qubits': num_qubits,
            'num_parameters': num_parameters,
            'runtime': runtime,
            'status': status,
            'energy': energy
        })
        
        print(f"Status: {status}")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Number of qubits: {num_qubits}")
        print(f"Number of parameters: {num_parameters}")
        
        # Stop if we hit a timeout
        if status == "Timeout":
            break
    
    plot_vqe_runtime(results)
    return results

def compare_classical_quantum(timeout_minutes=5, examples_per_qubit=2):
    """Compare classical and quantum solutions for sequences ordered by complexity."""
    print("\n=== Classical vs Quantum Comparison ===")
    
    # Generate and sort sequences
    sequences = generate_rna_dataset(examples_per_qubit)
    results = []
    timeout_seconds = timeout_minutes * 60
    
    for seq_data in sequences:
        sequence = seq_data['sequence']
        print(f"\nAnalyzing sequence: {sequence}")
        
        # Get classical solution
        classical_start = time.time()
        classical_solution, classical_energy = solve_qubo_with_cpsat(seq_data['L'], seq_data['Q'])
        classical_time = time.time() - classical_start
        
        # Get quantum solution
        qp = build_quadratic_program(seq_data['L'], seq_data['Q'])
        try:
            with timeout(timeout_seconds):
                quantum_start = time.time()
                result = solve_with_vqe(qp)
                quantum_time = time.time() - quantum_start
                quantum_energy = result.fval
                status = "Completed"
        except TimeoutException:
            quantum_time = timeout_seconds
            quantum_energy = None
            status = "Timeout"
        except Exception as e:
            quantum_time = time.time() - quantum_start
            quantum_energy = None
            status = f"Error: {str(e)}"
        
        results.append({
            'sequence': sequence,
            'length': seq_data['length'],
            'num_linear': seq_data['num_linear'],
            'classical_energy': classical_energy,
            'quantum_energy': quantum_energy,
            'classical_time': classical_time,
            'quantum_time': quantum_time,
            'status': status
        })
        
        print(f"Classical energy: {classical_energy:.2f}")
        print(f"Quantum energy: {quantum_energy if quantum_energy is not None else 'N/A'}")
        print(f"Classical time: {classical_time:.2f}s")
        print(f"Quantum time: {quantum_time:.2f}s")
        print(f"Status: {status}")
        
        # Stop if we hit a timeout
        if status == "Timeout":
            break
    
    plot_classical_quantum_comparison(results)
    return results

def compare_all_ansatz_vs_classical(timeout_minutes=5, examples_per_qubit=2):
    """Compare classical and all quantum ansatz solutions for sequences ordered by complexity."""
    print("\n=== Classical vs All Ansatz Comparison ===")
    from quantum_optimizer import get_ansatz
    ansatz_types = ["RealAmplitudes", "TwoLocal", "EfficientSU2"]
    ansatz_colors = {"RealAmplitudes": "purple", "TwoLocal": "blue", "EfficientSU2": "orange"}
    ansatz_markers = {"RealAmplitudes": "s", "TwoLocal": "^", "EfficientSU2": "D"}

    # Generate and sort sequences
    sequences = generate_rna_dataset(examples_per_qubit)
    results = []
    timeout_seconds = int(timeout_minutes * 60)

    for seq_data in sequences:
        sequence = seq_data['sequence']
        print(f"\nAnalyzing sequence: {sequence}")

        # Get classical solution
        classical_start = time.time()
        classical_solution, classical_energy = solve_qubo_with_cpsat(seq_data['L'], seq_data['Q'])
        classical_time = time.time() - classical_start

        quantum_energies = {}
        quantum_times = {}
        quantum_statuses = {}
        timeout_occurred = False
        
        for ansatz_type in ansatz_types:
            qp = build_quadratic_program(seq_data['L'], seq_data['Q'])
            quantum_start = time.time()
            try:
                with timeout(timeout_seconds):
                    result = solve_with_vqe(qp, ansatz_type=ansatz_type)
                    quantum_time = time.time() - quantum_start
                    quantum_energy = result.fval
                    status = "Completed"
            except TimeoutException:
                quantum_time = timeout_seconds
                quantum_energy = None
                status = "Timeout"
                timeout_occurred = True
            except Exception as e:
                quantum_time = time.time() - quantum_start
                quantum_energy = None
                status = f"Error: {str(e)}"
            quantum_energies[ansatz_type] = quantum_energy
            quantum_times[ansatz_type] = quantum_time
            quantum_statuses[ansatz_type] = status

        # Find closest quantum result to classical
        min_delta = None
        closest_ansatz = None
        for ansatz_type in ansatz_types:
            qe = quantum_energies[ansatz_type]
            if qe is not None:
                delta = abs(qe - classical_energy)
                if min_delta is None or delta < min_delta:
                    min_delta = delta
                    closest_ansatz = ansatz_type

        results.append({
            'sequence': sequence,
            'num_qubits': seq_data['num_linear'],
            'classical_energy': classical_energy,
            'quantum_energies': quantum_energies,
            'closest_ansatz': closest_ansatz,
            'min_delta': min_delta
        })

        print(f"Classical energy: {classical_energy:.2f}")
        for ansatz_type in ansatz_types:
            qe = quantum_energies[ansatz_type]
            print(f"{ansatz_type} energy: {qe if qe is not None else 'N/A'}")
        print(f"Closest ansatz: {closest_ansatz}, Δ = {min_delta:.2f}" if closest_ansatz else "No quantum result.")

        # Stop if any ansatz hit the timeout
        if timeout_occurred:
            print("\nStopping analysis due to timeout in one of the ansatz.")
            break

    plot_all_ansatz_vs_classical(results, ansatz_types, ansatz_colors, ansatz_markers)
    return results

def plot_qubo_scaling(results):
    """Plot the scaling of QUBO terms with sequence length, filtering out zero/negative values for log scale."""
    plt.figure(figsize=(15, 8))

    lengths = np.array([r['length'] for r in results])
    linear_terms = np.array([r['num_linear'] for r in results])
    quadratic_terms = np.array([r['num_quadratic'] for r in results])

    # Filter out zero or negative values for log scale
    valid_linear = linear_terms > 0
    valid_quadratic = quadratic_terms > 0
    valid = valid_linear & valid_quadratic

    lengths = lengths[valid]
    linear_terms = linear_terms[valid]
    quadratic_terms = quadratic_terms[valid]

    plt.plot(lengths, linear_terms, 'b-', marker='o', label='Linear terms')
    plt.plot(lengths, quadratic_terms, 'r-', marker='s', label='Quadratic terms')

    plt.xlabel('Sequence Length')
    plt.ylabel('Number of Terms')
    plt.title('Scaling of QUBO Terms with Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization

    save_plot(plt, "qubo_resource_scaling")
    plt.close()

def plot_vqe_runtime(results):
    """Create 2D plot of VQE runtime vs qubits with parameter annotations."""
    plt.figure(figsize=(15, 10))
    
    qubits = [r['num_qubits'] for r in results]
    parameters = [r['num_parameters'] for r in results]
    runtimes = [r['runtime'] for r in results]
    statuses = [r['status'] for r in results]
    
    # Separate points by status
    completed = [(q, t, p) for q, t, p, s in zip(qubits, runtimes, parameters, statuses) 
                if s == "Completed"]
    timeouts = [(q, t, p) for q, t, p, s in zip(qubits, runtimes, parameters, statuses) 
                if s == "Timeout"]
    errors = [(q, t, p) for q, t, p, s in zip(qubits, runtimes, parameters, statuses) 
              if s.startswith("Error")]
    
    # Sort completed points by qubit count
    completed.sort(key=lambda x: x[0])
    
    if completed:
        # Plot points
        plt.scatter([x[0] for x in completed], [x[1] for x in completed],
                   c='purple', marker='o', label='Completed', s=100)
        # Add connecting dotted line
        plt.plot([x[0] for x in completed], [x[1] for x in completed],
                c='purple', linestyle=':', alpha=0.5)
        # Add parameter annotations
        for x, y, p in completed:
            plt.annotate(f'p={p}', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    if timeouts:
        plt.scatter([x[0] for x in timeouts], [x[1] for x in timeouts],
                   c='r', marker='s', label='Timeout', s=100)
        for x, y, p in timeouts:
            plt.annotate(f'p={p}', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    if errors:
        plt.scatter([x[0] for x in errors], [x[1] for x in errors],
                   c='orange', marker='x', label='Error', s=100)
        for x, y, p in errors:
            plt.annotate(f'p={p}', (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Runtime (seconds)')
    plt.title('VQE Runtime vs Number of Qubits\n(parameter count shown next to points)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_plot(plt, "vqe_runtime_2d")
    plt.close()

def plot_classical_quantum_comparison(results):
    """Plot comparison between classical and quantum solutions in the requested style."""
    plt.figure(figsize=(15, 12))

    x = range(len(results))
    classical_energies = [r['classical_energy'] for r in results]
    quantum_energies = [r['quantum_energy'] if r['quantum_energy'] is not None else float('nan') for r in results]
    num_qubits = [r['num_linear'] for r in results]  # Use number of linear terms as qubits

    # Calculate y-axis limits with padding
    valid_energies = [e for e in classical_energies + quantum_energies if not np.isnan(e)]
    y_min = min(valid_energies)
    y_max = max(valid_energies)
    y_range = y_max - y_min
    y_padding = y_range * 0.3

    # Plot classical solutions with red circles
    plt.scatter(x, classical_energies, s=100, label='Classical', color='red', marker='o', alpha=1.0, zorder=3)

    # Plot quantum solutions with purple squares
    plt.scatter(x, quantum_energies, s=100, label='Quantum (VQE)', color='purple', marker='s', alpha=0.6, zorder=4)

    # Add connecting lines and energy differences
    for i in range(len(results)):
        if not np.isnan(quantum_energies[i]):
            plt.plot([i, i], [classical_energies[i], quantum_energies[i]], color='gray', linestyle='--', alpha=0.5, zorder=2)
            diff = quantum_energies[i] - classical_energies[i]
            plt.text(i, max(quantum_energies[i], classical_energies[i]) + y_padding/15, f'Δ = {diff:.2f}', ha='center', fontsize=10)

    plt.xlabel('RNA Sequence & #qubits')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Classical vs Quantum Energy Solutions (Energy in kcal/mol)')

    # Create vertical sequence labels with #qubits in parentheses
    labels = [f"{r['sequence']}\n(#qubits={r['num_linear']})" for r in results]
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)

    plt.legend()
    plt.grid(True, alpha=0.3, zorder=1)
    plt.ylim(y_min - y_padding*0.1, y_max + y_padding)
    plt.tight_layout()
    save_plot(plt, "classical_quantum_comparison")
    plt.close()

def plot_all_ansatz_vs_classical(results, ansatz_types, ansatz_colors, ansatz_markers):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(18, 10))
    x = range(len(results))
    classical_energies = [r['classical_energy'] for r in results]
    num_qubits = [r['num_qubits'] for r in results]

    # Calculate y-axis limits with padding
    valid_energies = [e for e in classical_energies]
    for r in results:
        for ansatz_type in ansatz_types:
            if r['quantum_energies'][ansatz_type] is not None:
                valid_energies.append(r['quantum_energies'][ansatz_type])
    
    y_min = min(valid_energies)
    y_max = max(valid_energies)
    y_range = y_max - y_min
    y_padding = y_range * 0.3

    # Count how many times each ansatz was closest
    ansatz_counts = {ansatz: 0 for ansatz in ansatz_types}
    for r in results:
        if r['closest_ansatz']:
            ansatz_counts[r['closest_ansatz']] += 1
    
    # Sort ansatz types by their count (highest to lowest)
    sorted_ansatz = sorted(ansatz_types, key=lambda x: ansatz_counts[x], reverse=True)

    # Plot classical
    plt.scatter(x, classical_energies, s=120, label='Classical', color='green', marker='o', alpha=1.0, zorder=3)

    # Plot each ansatz
    for ansatz_type in sorted_ansatz:
        energies = [r['quantum_energies'][ansatz_type] if r['quantum_energies'][ansatz_type] is not None else np.nan for r in results]
        count = ansatz_counts[ansatz_type]
        plt.scatter(x, energies, s=100, 
                   label=f'{ansatz_type} (# of wins: {count})', 
                   color=ansatz_colors[ansatz_type], 
                   marker=ansatz_markers[ansatz_type], 
                   alpha=0.7, 
                   zorder=4)

    # Draw lines and annotate delta for closest ansatz
    for i, r in enumerate(results):
        ca = r['closest_ansatz']
        if ca and r['quantum_energies'][ca] is not None:
            plt.plot([i, i], [r['classical_energy'], r['quantum_energies'][ca]], color=ansatz_colors[ca], linestyle='--', alpha=0.5, zorder=2)
            
            # Find the maximum value among all ansatz for this sequence
            max_ansatz_value = float('-inf')
            max_ansatz_type = None
            for ansatz_type in ansatz_types:
                if r['quantum_energies'][ansatz_type] is not None:
                    if r['quantum_energies'][ansatz_type] > max_ansatz_value:
                        max_ansatz_value = r['quantum_energies'][ansatz_type]
                        max_ansatz_type = ansatz_type
            
            # Position delta text above the highest ansatz value
            y_pos = max(r['classical_energy'], max_ansatz_value) + y_padding/4
            plt.text(i, y_pos, f'Δ={r["min_delta"]:.2f}', ha='center', fontsize=10, color=ansatz_colors[ca])

    # X-tick labels with rotation
    labels = [f"{r['sequence']}\n(#qubits={r['num_qubits']})" for r in results]
    plt.xticks(x, labels, rotation=45, ha='right', va='top')
    plt.xlabel('RNA Sequence')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Classical vs Quantum (All Ansatz) Energy Solutions')
    plt.legend()
    plt.grid(True, alpha=0.3, zorder=1)
    plt.ylim(y_min - y_padding*0.1, y_max + y_padding)
    plt.tight_layout()
    save_plot(plt, "all_ansatz_vs_classical")
    plt.close()

def main():
    """Run the analysis experiments."""
    print("Starting RNA Folding Analysis...")
    print(f"Results will be saved in: {os.path.abspath(RESULTS_DIR)}")
    
    # QUBO resource analysis up to length 100
    # print("\n=== QUBO Resource Analysis ===")
    # qubo_results = analyze_qubo_resources(max_length=100, step_size=1)
    
    # # VQE runtime analysis
    # print("\n=== VQE Runtime Analysis ===")
    # vqe_results = analyze_vqe_runtime(timeout_minutes=10)  # Changed to 10 minutes
    
    # # Classical vs Quantum comparison
    # print("\n=== Classical vs Quantum Comparison ===")
    # comparison_results = compare_classical_quantum(timeout_minutes=1)  # Changed to 10 minutes
    
    # Classical vs All Ansatz comparison
    print("\n=== Classical vs All Ansatz Comparison ===")
    all_ansatz_results = compare_all_ansatz_vs_classical(timeout_minutes=3, examples_per_qubit=1)  # 6 seconds
    
    print("\nAnalysis completed. Check results directory for output files.")

if __name__ == "__main__":
    main() 