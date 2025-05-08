from ortools.sat.python import cp_model

def solve_qubo_with_cpsat(L, Q):
    # Convert string keys to integers
    scale = 1000
    L_int = {int(k): int(v * scale) for k, v in L.items()}
    Q_int = {(int(i), int(j)): int(v * scale) for (i, j), v in Q.items()}

    print(f"[DEBUG] Number of variables: {len(L_int)}")
    print(f"[DEBUG] Number of quadratic terms: {len(Q_int)}")
    print(f"[DEBUG] Sample linear terms: {list(L_int.items())[:5]}")
    print(f"[DEBUG] Sample quadratic terms: {list(Q_int.items())[:5]}")

    model = cp_model.CpModel()
    n = max(
        max(L_int.keys(), default=0),
        max((max(pair) for pair in Q_int.keys()), default=0)
    ) + 1
    x = [model.NewBoolVar(f'x_{i}') for i in range(n)]

    # Objective: sum of linear and quadratic terms
    objective_terms = []
    for i, bias in L_int.items():
        objective_terms.append(bias * x[i])

    # For quadratic terms, introduce auxiliary variables
    for (i, j), bias in Q_int.items():
        if i != j:
            y_ij = model.NewBoolVar(f'y_{i}_{j}')
            # y_ij == x[i] AND x[j]
            model.AddMultiplicationEquality(y_ij, [x[i], x[j]])
            objective_terms.append(bias * y_ij)
        else:
            objective_terms.append(bias * x[i])

    print("[DEBUG] Starting solver...")
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = {i: int(solver.Value(x[i])) for i in range(n)}
        energy = solver.ObjectiveValue() / scale
        print("[DEBUG] Solver finished with a solution.")
        print("Best solution:", solution)
        print("Best energy:", energy)
        return solution, energy
    else:
        print("[DEBUG] No solution found.")
        return None, None