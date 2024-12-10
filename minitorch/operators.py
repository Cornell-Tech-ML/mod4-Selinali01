"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, TypeVar


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x (float): The first number
        y (float): The second number

    Returns:
    -------
        float: The product of x and y

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x (float): The number to return.

    Returns:
    -------
        float: The input number x.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x (float): The first number
        y (float): The second number

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x (float): The number to negate.

    Returns:
    -------
        float: The negated number.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Compare two numbers.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Compare two numbers.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        float: The maximum of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close.

    Args:
    ----
        x (float): The first number to compare.
        y (float): The second number to compare.

    Returns:
    -------
        bool: True if x and y are close, False otherwise.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid of a number.

    Args:
    ----
        x (float): The number to compute the sigmoid of.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU of a number.

    Args:
    ----
        x (float): The number to compute the ReLU of.

    Returns:
    -------
        float: The ReLU of x.

    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of a number.

    Args:
    ----
        x (float): The number to compute the logarithm of.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential of a number.

    Args:
    ----
        x (float): The number to compute the exponential of.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the logarithm function.

    Args:
    ----
        x (float): The input value from the forward pass.
        d (float): The gradient from the next layer.

    Returns:
    -------
        float: The gradient of the logarithm function.

    Raises:
    ------
        ValueError: If x is non-positive.

    """
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the inverse of a number.

    Args:
    ----
        x (float): The number to compute the inverse of.

    Returns:
    -------
        float: The inverse of x.

    """
    return 1.0 / x


def inv_back(x: float, grad: float) -> float:
    """Compute the gradient of the inverse function.

    Args:
    ----
        x (float): The input value from the forward pass.
        grad (float): The gradient from the next layer.

    Returns:
    -------
        float: The gradient of the inverse function.

    """
    return (-1.0 / x**2) * grad


def relu_back(x: float, grad: float) -> float:
    """Compute the derivative of the ReLU function.

    Args:
    ----
        x (float): The input value from the forward pass.
        grad (float): The gradient from the next layer.

    Returns:
    -------
        float: The gradient of the ReLU function.

    """
    return grad if x > 0 else 0.0


# ## Task 0.3
# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element in an iterable.

    Args:
    ----
        fn: Function from one value to one value

    Returns:
    -------
        A new iterable containing the results of applying f to each element in lst.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        lst (Iterable[float]): The input list of numbers.

    Returns:
    -------
        Iterable[float]: A new iterable with all elements negated.

    """
    return map(neg)(lst)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to pairs of elements from two iterables.

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        FUnction that takes two equally sized lists, produce new list applying fn(x,y) on each pair of elements

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        lst1 (Iterable[float]): The first input list of numbers.
        lst2 (Iterable[float]): The second input list of numbers.

    Returns:
    -------
        Iterable[float]: A new iterable with sums of corresponding elements.

    """
    return zipWith(add)(lst1, lst2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value by repeatedly applying a binary function.

    Args:
    ----
        fn: combine two values
        start: start value x_0

    Returns:
    -------
       FUnction that takes a list 'ls' of elements and computes reduction

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        lst (Iterable[float]): The input list of numbers.

    Returns:
    -------
        float: The sum of all elements in the list.

    """
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        lst (Iterable[float]): The input list of numbers.

    Returns:
    -------
        float: The product of all elements in the list.

    """
    return reduce(mul, 1.0)(lst)
