import numpy as np

INF = 10e9

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

    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("ERROR: Constant vector incorrectly sized")
        return

    n = len(b_matrix)
    x_old = np.zeros(n)
    x_new = np.zeros(n)
    new_line = "\n"
    augmented_matrix = np.concatenate((a_matrix, b_matrix), axis=1, dtype=float)
    print(f"Initial augmented matrix: {new_line}{augmented_matrix}")
    print("Solving:")

    error = np.zeros((n,1))
    continue_loop = True
    while continue_loop:
        x_new = np.zeros(n)
        num_iteration += 1
        for row in range(n):
            for j in range(n):
                if j != row:
                    x_new[row] += augmented_matrix[row, j] * x_old[j]
            x_new[row] = (augmented_matrix[row, n] - x_new[row]) / augmented_matrix[row, row]
        x_old = x_new.copy()

        # Calculate error
        error = np.abs(np.dot(a_matrix, x_new.reshape(-1, 1)) - b_matrix)
        if (error < epsilon).all():
            continue_loop = False

    print("Number of iterations: ",num_iteration)
    print("Solution of system:")
    for i in range(n):
        print(f"x{i + 1} = {x_new[i]:.2f}")

    error_print = error.reshape(-1)
    print(f"Error of Jacobi method when use epsilon = {epsilon:.2e}:")
    for i in range(n):
        print(f"Equation {i + 1}: error = {error_print[i]:.2e}")


def generate_DD_matrix(n):
    # Generate a random matrix
    matrix = np.random.rand(n, n)

    # Make it diagonally dominant
    for i in range(n):
        diagonal_element = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
        matrix[i, i] = diagonal_element + 1  # Add 1 to the diagonal element

    return matrix


def main():
    n = 200
    epsilon = 0.1
    variable_matrix = generate_DD_matrix(n)
    constant_matrix = np.random.rand(n, 1)
    jacobi(variable_matrix, constant_matrix, epsilon)


main()
