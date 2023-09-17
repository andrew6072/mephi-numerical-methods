import numpy as np

INF = 10e9

def to_DD(variable, constant):
    isDD = True
    n = len(variable)
    correct_pos = []
    for j in range(n):
        possible_index_swap = []
        for i in range(n):
            if (variable[i, j] >= (sum(variable[i]) - variable[i, j]) and i not in correct_pos): ## add and correct_pos[i]
                possible_index_swap.append(i)

        if (len(possible_index_swap) > 0): # if found possible row to swap this row
            index_max_diff = -1
            max_val = -INF
            for k in possible_index_swap: # find row with max variable[k, j] - (sum(variable[k]) - variable[k, j])
                if variable[k, j] - (sum(variable[k]) - variable[k, j]) > max_val:
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


def gauss_elimination(a_matrix, b_matrix):
    num_iteration = 0

    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("ERROR: Square matrix is not given")
        return

    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("ERROR: Constant vector incorrectly sized")
        return

    n = len(b_matrix)
    m = n - 1
    i = 0
    # j = i - 1
    x = np.zeros(n)
    new_line = "\n"

    # Create augmented matrix
    augmented_matrix = np.concatenate((a_matrix, b_matrix), axis=1, dtype=float)
    print(f"Initial augmented matrix: {new_line}{augmented_matrix}")
    print("Solving:")

    while i < n - 1:
        # Partial pivoting
        for p in range(i + 1, n):
            if abs(augmented_matrix[i, i]) < abs(augmented_matrix[p, i]):
                augmented_matrix[[p, i]] = augmented_matrix[[i, p]]

        if augmented_matrix[i, i] == 0.0:
            print("ERROR: Divide by 0")
            return

        for j in range(i + 1, n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i, i]
            augmented_matrix[j] -= scaling_factor * augmented_matrix[i]
            # print(augmented_matrix)
            num_iteration += 1

        i += 1

    print("Number of iteration: ", num_iteration)
    print("            n = ", n)

    # Backward substitution
    x[m] = augmented_matrix[m, n] / augmented_matrix[m, m]

    for k in range(n - 2, -1, -1):
        x[k] = augmented_matrix[k, n]
        for j in range(k + 1, n):
            x[k] -= augmented_matrix[k][j] * x[j]
        x[k] /= augmented_matrix[k, k]

    # for number, value in enumerate(x):
    #     print(f"x{number} = {value:.10f}")

    return x


def solve(A, B):
    x = gauss_elimination(A, B)
    x0 = np.linalg.solve(A, B).flatten()

    error = np.sqrt(np.sum((x - x0) ** 2))
    print(f"        Error = {error:.2e}")

    incoherence_g = np.sqrt(np.sum((np.dot(A, np.transpose(x)) - B.flatten()) ** 2))
    incoherence_0 = np.sqrt(np.sum((np.dot(A, x0) - B.flatten()) ** 2))

    print(f"incoherence_g = {incoherence_g:.2e}")
    print(f"incoherence_0 = {incoherence_0:.2e}")


# variable_matrix = np.array([[1, 1, 3], [0, 1, 3], [-1, 3, 0]])
# constant_matrix = np.array([[1], [3], [5]])

variable_matrix = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9, 3, 1, 0, 0, 0],
        [9, 3, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 25, 5, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 25, 5, 1],
        [0, 0, 0, 0, 0, 0, 64, 8, 1],
        [6, 1, 0, -6, -1, 0, 0, 0, 0],
        [0, 0, 0, 10, 1, 0, -10, -1, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
constant_matrix = np.array([[2], [3], [3], [9], [9], [10], [0], [0], [0]])

isDD = to_DD(variable_matrix, constant_matrix)
print("Matrix is DD: ",isDD)


# gauss_elimination(variable_matrix, constant_matrix)
#
# x0 = np.linalg.solve(variable_matrix, constant_matrix)
# print(x0.flatten())

# variable_matrix = np.random.rand(200, 200)
# constant_matrix = np.random.rand(200, 1)


solve(variable_matrix, constant_matrix)
