"""
Egyptian algorithm
"""

def isodd(n):
    """
    returns True if n is odd
    """
    return n & 0x1 == 1

def egyptian_multiplication(a, n):
    """
    returns the product a * n

    assume n is a nonegative integer
    """
    if n == 1:
        return a
    if n == 0:
        return 0

    if isodd(n):
        return egyptian_multiplication(a + a, n // 2) + a
    else:
        return egyptian_multiplication(a + a, n // 2)


if __name__ == '__main__':
    # this code runs when executed as a script
    print("Examples of egyptian_multiplication function:")
    for a in [1,2,3]:
        for n in [1,2,5,10]:
            print("{} * {} = {}".format(a, n, egyptian_multiplication(a,n)))


def power(a, n):
    """
    computes the power a ** n

    assume n is a nonegative integer
    """

    if n == 1:
        return a
    if n == 0:
        return 1

    if isodd(n):
        return power(a * a, n // 2) * a
    else:
        return power(a * a, n // 2)

if __name__ == '__main__':
    # this code runs when executed as a script
    print("Examples of power function:")
    a = 3; n = 3
    print("{} * {} = {}".format(a, n, power(a,n)))
    a = 4; n = 4
    print("{} * {} = {}".format(a, n, power(a,n)))
    a = 5; n = 3
    print("{} * {} = {}".format(a, n, power(a,n)))
