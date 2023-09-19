import numpy as np

INF = 10e9


def norm(vec):
    return np.max(np.abs(vec))


def check_DD(matrix):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    n = matrix.shape[0]

    for i in range(n):
        diagonal_element = abs(matrix[i, i])
        row_sum = np.sum(np.abs(matrix[i, :])) - diagonal_element

        if diagonal_element <= row_sum:
            return False

    return True


def to_DD(variable, constant): # convert a matrix to DD form
    isDD = check_DD(variable)
    if isDD:
        return True
    n = len(variable)
    correct_pos = []
    for j in range(n):
        possible_index_swap = []
        for i in range(n):
            if (np.abs(variable[i, j]) >= (np.sum(np.abs(variable[i])) - np.abs(variable[i, j]))) and i not in correct_pos: ## add and correct_pos[i]
                possible_index_swap.append(i)

        if (len(possible_index_swap) > 0): # if found possible row to swap this row
            index_max_diff = -1
            max_val = -INF
            for k in possible_index_swap: # find row with max variable[k, j] - (sum(variable[k]) - variable[k, j])
                if (np.abs(variable[k, j]) >= (np.sum(np.abs(variable[k])) - np.abs(variable[k, j]))) > max_val:
                    max_val = variable[k, j] - (sum(variable[k]) - variable[k, j])
                    index_max_diff = k
            # swap row index_max_diff with row j then update to correct pos
            if j != index_max_diff:
                variable[[index_max_diff, j]] = variable[[j, index_max_diff]]
                constant[[index_max_diff, j]] = constant[[j, index_max_diff]]
            correct_pos.append(j)
        else:
            isDD = False
    return isDD


def jacobi(a_matrix, b_matrix, epsilon):
    # try converting matrix to DD form
    isDD = to_DD(a_matrix, b_matrix)

    print(f"System is Diagonally Dominant: {isDD}")

    if not isDD:
        return

    num_iteration = 0

    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("ERROR: Square matrix is not given")
        return

    if b_matrix.shape[0] != a_matrix.shape[0]:
        print("ERROR: Constant vector incorrectly sized")
        return

    n = len(b_matrix)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    error = 1.0e10
    while error > epsilon:
        x_new = np.zeros(n)
        num_iteration += 1
        for row in range(n):
            x_new[row] = np.sum(a_matrix[row] * x_old) - a_matrix[row, row] * x_old[row]

            x_new[row] = (b_matrix[row] - x_new[row]) / a_matrix[row, row]

        error = norm(a_matrix @ x_new - b_matrix)

        x_old = x_new.copy()

    return x_new, num_iteration


def generate_DD_matrix(n):
    # Generate a random matrix
    matrix = np.random.rand(n, n)

    # Make it diagonally dominant
    for i in range(n):
        diagonal_element = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
        matrix[i, i] = diagonal_element + 1.0  # Add 1 to the diagonal element

    return matrix


def main():
    n = 20
    epsilon = 1.0e-10
    a_matrix = generate_DD_matrix(n)
    b_matrix = np.random.rand(n)
    x_new, num_iteration = jacobi(a_matrix, b_matrix, epsilon)

    print("Result:")
    print("Number of iterations: ", num_iteration)

    x0 = np.linalg.solve(a_matrix, b_matrix)
    error_val = norm(x_new - x0)
    print(f"        Error = {error_val:.2e}")

    incoherence_g = norm(a_matrix @ x_new - b_matrix)
    incoherence_0 = norm(a_matrix @ x0 - b_matrix)

    print(f"incoherence_g = {incoherence_g:.2e}")
    print(f"incoherence_0 = {incoherence_0:.2e}")


main()
