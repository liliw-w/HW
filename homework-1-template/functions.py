"""
A library of functions
"""
import numpy as np
import matplotlib.pyplot as plt
import numbers

class AbstractFunction:
    """
    An abstract function class
    """

    def __str__(self):
        return "AbstractFunction"


    def __repr__(self):
        return "AbstractFunction"


    def evaluate(self, x):
        """
        evaluate at x

        assumes x is a numeric value, or numpy array of values
        """
        raise NotImplementedError("evaluate")


    def derivative(self):
        """
        returns another function f' which is the derivative of x
        """
        raise NotImplementedError("derivative")


    def __call__(self, x):
        """
        if x is another AbstractFunction, return the composition of functions

        if x is a string return a string that uses x as the indeterminate

        otherwise, evaluate function at a point x using evaluate
        """
        if isinstance(x, AbstractFunction):
            return Compose(self, x)
        elif isinstance(x, str):
            return self.__str__().format(x)
        else:
            return self.evaluate(x)


    # the rest of these methods will be implemented when we write the appropriate functions
    def __add__(self, other):
        """
        returns a new function expressing the sum of two functions
        """
        return Sum(self, other)


    def __mul__(self, other):
        """
        returns a new function expressing the product of two functions
        """
        return Product(self, other)


    def __neg__(self):
        return Scale(-1)(self)


    def __truediv__(self, other):
        return self * other**-1


    def __pow__(self, n):
        return Power(n)(self)


    def plot(self, vals=np.linspace(-1,1,100), **kwargs):
        """
        plots function on values
        pass kwargs to plotting function
        """
        plt.plot(vals, self.evaluate(vals), **kwargs)
        #plt.show()

        #raise NotImplementedError("plot")


    def taylor_series(self, x0, deg=5):
        """
        Returns the Taylor series of f centered at x0 truncated to degree k.
        """
        s = Constant(self.evaluate(x0))

        if deg < 1:
            return s
        for i in np.arange(1, deg+1):
            self = (Constant(1/i) * self).derivative()
            s = s + Constant(self.evaluate(x0)) * Power(i)( Polynomial(1, -1*x0) )
        return s


class Polynomial(AbstractFunction):
    """
    polynomial c_n x^n + ... + c_1 x + c_0
    """

    def __init__(self, *args):
        """
        Polynomial(c_n ... c_0)

        Creates a polynomial
        c_n x^n + c_{n-1} x^{n-1} + ... + c_0
        """
        self.coeff = np.array(list(args))


    def __repr__(self):
        return "Polynomial{}".format(tuple(self.coeff))


    def __str__(self):
        """
        We'll create a string starting with leading term first

        there are a lot of branch conditions to make everything look pretty
        """
        s = ""
        deg = self.degree()
        for i, c in enumerate(self.coeff):
            if i < deg-1:
                if c == 0:
                    # don't print term at all
                    continue
                elif c == 1:
                    # supress coefficient
                    s = s + "({{0}})^{} + ".format(deg - i)
                else:
                    # print coefficient
                    s = s + "{}({{0}})^{} + ".format(c, deg - i)
            elif i == deg-1:
                # linear term
                if c == 0:
                    continue
                elif c == 1:
                    # suppress coefficient
                    s = s + "{0} + "
                else:
                    s = s + "{}({{0}}) + ".format(c)
            else:
                if c == 0 and len(s) > 0:
                    continue
                else:
                    # constant term
                    s = s + "{}".format(c)

        # handle possible trailing +
        if s[-3:] == " + ":
            s = s[:-3]

        return s


    def evaluate(self, x):
        """
        evaluate polynomial at x
        """
        if isinstance(x, numbers.Number):
            ret = 0
            for k, c in enumerate(reversed(self.coeff)):
                ret = ret + c * x**k
            return ret
        elif isinstance(x, np.ndarray):
            x = np.array(x)
            # use vandermonde matrix
            return np.vander(x, len(self.coeff)).dot(self.coeff)


    def derivative(self):
        if len(self.coeff) == 1:
            return Polynomial(0)
        return Polynomial(*(self.coeff[:-1] * np.array([n+1 for n in reversed(range(self.degree()))])))


    def degree(self):
        return len(self.coeff) - 1


    def __add__(self, other):
        """
        Polynomials are closed under addition - implement special rule
        """
        if isinstance(other, Polynomial):
            # add
            if self.degree() > other.degree():
                coeff = self.coeff
                coeff[-(other.degree() + 1):] += other.coeff
                return Polynomial(*coeff)
            else:
                coeff = other.coeff
                coeff[-(self.degree() + 1):] += self.coeff
                return Polynomial(*coeff)

        else:
            # do default add
            return super().__add__(other)


    def __mul__(self, other):
        """
        Polynomials are clused under multiplication - implement special rule
        """
        if isinstance(other, Polynomial):
            return Polynomial(*np.polymul(self.coeff, other.coeff))
        else:
            return super().__mul__(other)


class Affine(Polynomial):
    """
    affine function a * x + b
    """
    def __init__(self, a, b):
        super().__init__(a, b)


class Scale(Polynomial):
    def __init__(self, a):
        super().__init__(a, 0)


class Constant(Polynomial):
    def __init__(self, c):
        super().__init__(c)


class Compose(AbstractFunction):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __str__(self):
        # From Answer: return "{}".format(self.f).format(self.g)
        return self.f.__str__().format(self.g.__str__())

    def __repr__(self):
        # From Answer: return "Compose({}, {})".format(self.f.__repr__(), self.g.__repr__())
        return "Compose({f}, {g})".format(f = self.f.__repr__(), g = self.g.__repr__())

    def evaluate(self, x):
        # From Answer: return self.f(self.g(x))
        return self.f.evaluate( self.g.evaluate(x) )

    def derivative(self):
        return self.f.derivative()(self.g) * self.g.derivative()


class Product(AbstractFunction):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __str__(self):
        # From Answer: return "({}) * ({})".format(self.f, self.g)
        return "(" + self.f.__str__() + ") * (" + self.g.__str__() + ")"

    def __repr__(self):
        return "Product({f}, {g})".format(f = self.f.__repr__(), g = self.g.__repr__())

    def evaluate(self, x):
        return self.f.evaluate(x) * self.g.evaluate(x)

    def derivative(self):
        return self.f.derivative()*self.g + self.f*self.g.derivative()


class Sum(AbstractFunction):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __str__(self):
        # From Answer: return "{} + {}".format(self.f, self.g)
        return self.f.__str__() + " + " + self.g.__str__()

    def __repr__(self):
        return "Sum({f}, {g})".format(f = self.f.__repr__(), g = self.g.__repr__())

    def evaluate(self, x):
        # From Answer: return self.f(x) + self.g(x)
        return self.f.evaluate(x) + self.g.evaluate(x)

    def derivative(self):
        # From Answer: return Sum(self.f.derivative(), self.g.derivative())
        return self.f.derivative() + self.g.derivative()


class Power(AbstractFunction):
    def __init__(self, n):
        self.power = n

    def __repr__(self):
        return "Power({})".format(self.power)

    def __str__(self):
        if self.power == 1:
            return "{0}"
        return "({{0}})^{}".format(self.power)

    def evaluate(self, x):
        return np.power(x, self.power)

    def derivative(self):
        # Use compose class
        return Scale(self.power)( Power(self.power-1) )

        #if self.power == 0:
        #    return 0
        # return Polynomial(*( np.insert(np.zeros(self.power-1), 0, self.power) ))

class Log(AbstractFunction):
    def __init__(self):
        pass

    def __repr__(self):
        return "Log()"

    def __str__(self):
        return "Log({0})"

    def evaluate(self, x):
        x = np.array(x)
        if np.all(x>0):
            return np.log(x)
        else:
            raise Exception("Log() only function on positive numbers!")

    def derivative(self):
        return Power(-1)


class Exponential(AbstractFunction):
    def __init__(self):
        pass

    def __repr__(self):
        return "Exponential()"

    def __str__(self):
        return "exp({0})"

    def evaluate(self, x):
        return np.exp(x)

    def derivative(self):
        return Exponential()


class Sin(AbstractFunction):
    def __init__(self, coef = 1):
        self.coef = coef

    def __repr__(self):
        if self.coef == 1:
            return "Sin()"
        return "{}Sin()".format(self.coef)

    def __str__(self):
        if self.coef == 1:
            return "Sin({0})"
        return "{}Sin({{0}})".format(self.coef)

    def evaluate(self, x):
        return np.sin(x) * self.coef

    def derivative(self):
        return Cos(self.coef)


class Cos(AbstractFunction):
    def __init__(self, coef = 1):
        self.coef = coef

    def __repr__(self):
        if self.coef == 1:
            return "Cos()"
        return "{}Cos()".format(self.coef)

    def __str__(self):
        if self.coef == 1:
            return "Cos({0})"
        return "{}Cos({{0}})".format(self.coef)

    def evaluate(self, x):
        return np.cos(x) * self.coef

    def derivative(self):
        # From Answer: return Constant(-1) * Sin() (In this way, no need to use self.coef)
        return Sin(-1 * self.coef)


class Symbolic(AbstractFunction):
    def __init__(self, f):
        self.f = f

    def __repr__(self):
        return "Symbolic()"

    def __str__(self):
        return "{}({{0}})".format(self.f)

    def evaluate(self, x):
        # return super().__call__(str(x))
        return self.__str__().format(x)

    def derivative(self):
        return Symbolic(self.f + '\'')


def newton_root(f, x0, tol=1e-8):
    """
    find a point x so that f(x) is close to 0,
    measured by abs(f(x)) < tol

    Use Newton's method starting at point x0
    """
    if not(isinstance(f, AbstractFunction)) or isinstance(f, Symbolic):
        raise Exception("Function f should be an AbstractFunction but not Symbolic!")

    eps = 10
    while eps > tol:
        x1 = x0 - f(x0)/f.derivative()(x0)
        x0 = x1
        eps = abs(f(x0))

    return x0


def newton_extremum(f, x0, tol=1e-8):
    """
    find a point x which is close to a local maximum or minimum of f,
    measured by abs(f'(x)) < tol

    Use Newton's method starting at point x0
    """
    if not(isinstance(f, AbstractFunction)) or isinstance(f, Symbolic):
        raise Exception("Function f should be an AbstractFunction but not Symbolic!")

    return newton_root(f.derivative(), x0=x0, tol=tol)


if __name__ == '__main__':
    p = Polynomial(5,3,1)
    p.plot(color='red')

