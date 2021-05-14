"""
fizzbuzz.

Write a python script which prints the numbers from 1 to 100,
but for multiples of 3 print "fizz" instead of the number,
for multiples of 5 print "buzz" instead of the number,
and for multiples of both 3 and 5 print "fizzbuzz" instead of the number.
"""


def print_special(seq):
    """Print numbers and words."""
    for i in seq:
        if i % 3 != 0 and i % 5 != 0:
            print(i)
        elif i % 3 == 0 and i % 5 != 0:
            print('fizz')
        elif i % 3 != 0 and i % 5 == 0:
            print('buzz')
        else:
            print('fizzbuzz')


if __name__ == '__main__':
    seq = list(range(1, 101))
    print_special(seq)
