import random
import time
import os
import sys

def naive_multiply(m1, m2):
    n = len(m1)
    m3 = [[0] * n for _ in range(n)]
    for i in range(n): # Optimized loop order
        for k in range(n):
            for j in range(n):
                m3[i][j] += m1[i][k] * m2[k][j]
    return m3

def add(m1, m2):
    n = len(m1)
    return [[m1[i][j] + m2[i][j] for j in range(n)] for i in range(n)]

def subtract(m1, m2):
    n = len(m1)
    return [[m1[i][j] - m2[i][j] for j in range(n)] for i in range(n)]

def strassen_multiply(m1, m2, crossover):
    n = len(m1)

    # Base case
    if n <= crossover:
        return naive_multiply(m1, m2)
    
    # Pad if odd (make new copies)
    if n % 2 != 0:
        m1 = [row + [0] for row in m1] + [[0] * (n+1)]
        m2 = [row + [0] for row in m2] + [[0] * (n+1)]
    
    # Ceiling (if n is odd, goes up for mid, if n is even, preserves exact mid)
    mid = int((n+1)/2)

    # Split matrices
    A = [row[:mid] for row in m1[:mid]]
    B = [row[mid:] for row in m1[:mid]]
    C = [row[:mid] for row in m1[mid:]]
    D = [row[mid:] for row in m1[mid:]]

    E = [row[:mid] for row in m2[:mid]]
    F = [row[mid:] for row in m2[:mid]]
    G = [row[:mid] for row in m2[mid:]]
    H = [row[mid:] for row in m2[mid:]]


    # Optimization for testing
    P1 = naive_multiply(A, subtract(F, H))
    P2 = naive_multiply(add(A, B), H)
    P3 = naive_multiply(add(C, D), E)
    P4 = naive_multiply(D, subtract(G, E))
    P5 = naive_multiply(add(A, D), add(E, H))
    P6 = naive_multiply(subtract(B, D), add(G, H))
    P7 = naive_multiply(subtract(C, A), add(E, F))


    # Compute 7 products

    # P1 = strassen_multiply(A, subtract(F, H), crossover)
    # P2 = strassen_multiply(add(A, B), H, crossover)
    # P3 = strassen_multiply(add(C, D), E, crossover)
    # P4 = strassen_multiply(D, subtract(G, E), crossover)
    # P5 = strassen_multiply(add(A, D), add(E, H), crossover)
    # P6 = strassen_multiply(subtract(B, D), add(G, H), crossover)
    # P7 = strassen_multiply(subtract(C, A), add(E, F), crossover)


    # Find 4 quadrants of A x B
    AEBG = add(subtract(P4, P2), add(P5, P6))
    AFBH = add(P1, P2)
    CEDG = add(P3, P4)
    CFDH = add(subtract(P1, P3), add(P5, P7))

    # Combine quadrants
    result = []

    for i in range(0, mid): # Top half of rows
        result.append(AEBG[i] + AFBH[i]) # Doesn't literally add ([a,b]+[c,d]=[a,b,c,d])
    for i in range(0, mid):
        result.append(CEDG[i] + CFDH[i]) # Bottom half of rows

    # Unpad if necessary
    if n % 2 != 0:
        result = [row[:n] for row in result[:n]]

    return result

# Matrix types for testing
t1 = [0, 1] # Sparse binary
t2 = [0, 1, 2] # Low-range integer
t3 = [-1, 0, 1] # Balanced sign
t4 = list(range(101)) # Dense random integer 

def s_tester(filename, type, dimension, crossover):
    with open(filename, "w") as f:
        for i in range(1, dimension + 1):
            total_time = 0
            for _ in range(5):  # 5 trials
                A = [[random.choice(type) for _ in range(i)] for _ in range(i)]
                B = [[random.choice(type) for _ in range(i)] for _ in range(i)]

                start = time.time()
                strassen_multiply(A, B, crossover)
                end = time.time()

                total_time += end - start

            avg_time = total_time / 5
            f.write(f"{i} {avg_time}\n")
    
def n_tester(filename, type, dimension):
    with open(filename, "w") as f:
        for i in range(1, dimension + 1):
            total_time = 0
            for _ in range(5):  # 5 trials
                A = [[random.choice(type) for _ in range(i)] for _ in range(i)]
                B = [[random.choice(type) for _ in range(i)] for _ in range(i)]

                start = time.time()
                naive_multiply(A, B)
                end = time.time()

                total_time += end - start

            avg_time = total_time / 5
            f.write(f"{i} {avg_time}\n")

def triangles():
    probabilities = [0.01, 0.02, 0.03, 0.04, 0.05]
    n = 1024

    for p in probabilities:
        A = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                if random.random() <= p:
                    A[i][j] = 1
                    A[j][i] = 1
        A_cubed = strassen_multiply(A, strassen_multiply(A, A, 1), 1)
        trace = 0
        for i in range(0, n):
            trace += A_cubed[i][i]
        print(f"p = {p}, no. of triangles = {trace / 6}")

def main():
    flag = sys.argv[1]
    dim = int(sys.argv[2])
    inputfile = sys.argv[3]

    A = [[0] * dim for _ in range(dim)]
    B = [[0] * dim for _ in range(dim)]

    # Autograder flag
    if flag == "0":
        with open(inputfile, "r") as f:

            # Read first d^2 elements as A
            for i in range(0, dim):
                for j in range(0, dim):
                    A[i][j] = int(f.readline())

            # Read second d^2 elements as B
            for i in range(0, dim):
                for j in range(0, dim):
                    B[i][j] = int(f.readline())
    
        product = strassen_multiply(A, B, 50) # Crossover set to 50 based on experimental results

        for i in range(0, dim):
            print(product[i][i]) 

    # Testing flagtO
    if flag == "1":
        # Strassens for n up to 100
        s_tester("strassen_t1.txt", t1, 100, 1)
        s_tester("strassen_t2.txt", t2, 100, 1)
        s_tester("strassen_t3.txt", t3, 100, 1)
        s_tester("strassen_t4.txt", t4, 100, 1)

        # Naive for n up to 100
        n_tester("naive_t1.txt", t1, 100)
        n_tester("naive_t2.txt", t2, 100)
        n_tester("naive_t3.txt", t3, 100)
        n_tester("naive_t4.txt", t4, 100)

if __name__ == "__main__":
        # Strassens for n up to 100
        s_tester("strassen_t1.txt", t1, 150, 1)
        s_tester("strassen_t2.txt", t2, 150, 1)
        s_tester("strassen_t3.txt", t3, 150, 1)
        s_tester("strassen_t4.txt", t4, 150, 1)

        # Naive for n up to 100
        n_tester("naive_t1.txt", t1, 150)
        n_tester("naive_t2.txt", t2, 150)
        n_tester("naive_t3.txt", t3, 150)
        n_tester("naive_t4.txt", t4, 150)