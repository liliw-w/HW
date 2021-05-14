"""
fibonacci

functions to compute fibonacci numbers

Complete problems 2 and 3 in this file.
"""

import time  # to compute runtimes
from tqdm import tqdm  # progress bar
from egyptian import isodd
import numpy as np
import matplotlib.pyplot as plt


# Question 2
def fibonacci_recursive(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    res = fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

    return res


# Question 2
def fibonacci_iter(n):
    if n == 0:
        return 0
    a = 0
    b = 1
    for i in range(n - 1):
        a, b = b, a + b
    return b


# Question 3
def fibonacci_power(n, dtype=None):
    """
    computes the power a ** n

    assume n is a nonegative integer
    """
    if n == 0:
        return 0

    def power(A, n):
        """
        computes the power A ** n

        assume n is a nonegative integer
        """
        if n == 0:
            return np.identity(2)
        if n == 1:
            return A

        if isodd(n):
            return power(A @ A, n // 2) @ A
        else:
            return power(A @ A, n // 2)

    A = np.array([1, 1, 1, 0], dtype=dtype).reshape(2, 2)
    Fn = power(A, n - 1) @ np.array([1, 0], dtype=dtype).reshape(2, 1)

    return int(Fn[0])

if __name__ == '__main__':
    """
    this section of the code only executes when
    this file is run as a script.
    """
    print("The first 30 Fibonacci numbers by fibonacci_recursive:", end='\n')
    print([fibonacci_recursive(n) for n in range(30)])
    print("The first 30 Fibonacci numbers by fibonacci_iter:", end='\n')
    print([fibonacci_iter(n) for n in range(30)])
    print("The first 30 Fibonacci numbers by fibonacci_power:", end='\n')
    print([fibonacci_power(n) for n in range(30)])

    try:
        print("Use np.float64")
        print(fibonacci_power(10 ** 13, dtype=np.float64))
    except ValueError as e:
        print(e)

    try:
        print("Use np.int64", end='\n')
        print(fibonacci_power(10 ** 13, dtype=np.int64))
    except ValueError as e:
        print(e)


    def get_runtimes(ns, f):
        """
        get runtimes for fibonacci(n)

        e.g.
        trecursive = get_runtimes(range(30), fibonacci_recusive)
        will get the time to compute each fibonacci number up to 29
        using fibonacci_recursive
        """
        ts = []
        for n in tqdm(ns):
            t0 = time.time()
            fn = f(n)
            t1 = time.time()
            ts.append(t1 - t0)

        return ts


    nrecursive = range(35)
    trecursive = get_runtimes(nrecursive, fibonacci_recursive)

    niter = range(1000)
    titer = get_runtimes(niter, fibonacci_iter)

    npower = range(1000)
    tpower = get_runtimes(npower, fibonacci_power)

    ## write your code for problem 4 below...
    plt.loglog(nrecursive, trecursive, label='recursive')
    plt.loglog(niter, titer, label='iter')
    plt.loglog(npower, tpower, label='power')

    plt.legend()
    plt.xlabel("log(n)")
    plt.ylabel("log scale")
    plt.title("Run time")
    plt.savefig('fibonacci_runtime.png')
