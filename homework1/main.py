import pathlib


def parse_coefficient(term: str, sign: str) -> float:
    result = 0.0
    if 'x' in term or 'y' in term or 'z' in term:
        coefficient = term[:-1]
        if sign == '+' or sign == '':
            result = 1.0 if coefficient == '' else float(coefficient)
        elif sign == '-':
            result = -1.0 if coefficient == '' else -1.0 * float(coefficient)
    return result


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []
    with path.open() as file:
        for line in file:
            left_side, right_side = line.split('=')
            right_side = right_side.strip()
            terms = left_side.split()
            a = parse_coefficient(terms[0], '')
            b = parse_coefficient(terms[2], terms[1])
            c = parse_coefficient(terms[4], terms[3])
            A.append([a, b, c])
            B.append(float(right_side))
    return A, B


def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - \
               matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + \
               matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])


def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def norm(vector: list[float]) -> float:
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix))]


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix)
    if det == 0:
        return []
    result = []
    for i in range(len(matrix)):
        new_matrix = [matrix[j].copy() for j in range(len(matrix))]
        for j in range(len(matrix)):
            new_matrix[j][i] = vector[j]
        result.append(determinant(new_matrix) / det)
    return result


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    return [[matrix[x][y] for y in range(len(matrix)) if y != j] for x in range(len(matrix)) if x != i]


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    return [[(-1) ** (i + j) * determinant(minor(matrix, i, j)) for j in range(len(matrix))] for i in range(len(matrix))]


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    adj = adjoint(matrix)
    det = determinant(matrix)
    if det == 0:
        return []
    return [sum(adj[i][j] * vector[j] for j in range(len(vector))) / det for i in range(len(matrix))]


A, B = load_system(pathlib.Path("file.txt"))
print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{minor(A, 0, 0)=}")
print(f"{minor(A, 0, 1)=}")
print(f"{minor(A, 0, 0)=}")
print(f"{cofactor(A)=}")
print(f"{solve(A, B)=}")
