################################################################################
# Project Euler Practice
# Author: kmishra9
# Previous Attempts: https://drive.google.com/file/d/0BwPzCEOJw0myMTl2STJaQjBoOGs/view
################################################################################

import os
import itertools
import numpy as np
import scipy as sp
from datetime import *
from typing import *
from functools import reduce
os.chdir("/Users/kunalmishra/Project-Euler/")

################################################################################
# 1
################################################################################

def multiples_of_3_or_5(n):
    return sum([i for i in range(n) if i % 3 == 0 or i % 5 == 0])

assert multiples_of_3_or_5(10) == 23
multiples_of_3_or_5(1000)

################################################################################
# 2
################################################################################

def even_fibonacci_numbers(threshold=4000000):
    # Calculate all fibonacci numbers <
    fibonaccis = [0, 1]
    next_fibonacci = sum(fibonaccis)
    while next_fibonacci <= threshold:
        next_fibonacci = fibonaccis[-1] + fibonaccis[-2]
        fibonaccis.append(next_fibonacci)

    # Calculate the sum of the even fibonaccis
    return sum([i for i in fibonaccis if i % 2 == 0])

def even_fibonacci_numbers_recursive(first=0, second=1, total=0, threshold=4000000):
    if first+second > threshold:
        return total
    return even_fibonacci_numbers_recursive(first=second,
                                            second=first+second,
                                            total=total+first+second if (first+second) % 2 == 0 else total,
                                            threshold=threshold)

assert even_fibonacci_numbers() == even_fibonacci_numbers_recursive()
even_fibonacci_numbers()

################################################################################
# 3
################################################################################

def is_factor(n, potential_factor):
    """Returns whether potential_factor is or is not a true factor of n"""
    return n % potential_factor == 0

def is_prime(n):
    for i in range(2, int(n**.5)+1):
        if is_factor(n=n, potential_factor=i):
            return False
    return True

def factor(n):
    for i in range(2, n+1):
        if is_factor(n=n, potential_factor=i):
            return (i, n//i)

def largest_prime_factor(n):
    if is_prime(n):
        return n

    factor_1, factor_2 = factor(n)

    if is_prime(factor_1) and is_prime(factor_2):
        return max(factor_1, factor_2)

    return max(largest_prime_factor(factor_1), largest_prime_factor(factor_2))

assert largest_prime_factor(10) == 5
assert largest_prime_factor(11) == 11
assert largest_prime_factor(13195) == 29
largest_prime_factor(600851475143)

################################################################################
# 4
################################################################################

def is_palindrome(n):
    assert type(n) is int
    return str(n) == str(n)[::-1]

def largest_palindrome_product(top, bottom):
    largest_so_far = -1
    for i in range(top, bottom, -1):
        for j in range(i, bottom, -1):
            product = i * j
            if product > largest_so_far and is_palindrome(product):
                largest_so_far = product
            elif product < largest_so_far:
                break

    return largest_so_far

assert largest_palindrome_product(top=99, bottom=10) == 9009
largest_palindrome_product(top=999, bottom=100)

################################################################################
# 5 - Attempt 1
################################################################################

def divisible_through(count, n):
    """Returns whether count is divisble by numbers 1 through n"""
    for i in range(n, 1, -1):
        if not is_factor(n=count, potential_factor=i):
            return False

    return True

def smallest_multiple_inefficient(n):
    """Finds the least common multiple of all numbers 1 through n"""
    count = 1
    while not divisible_through(count, n):
        count += 1

    return count

################################################################################
# 5 - Attempt 2
################################################################################

def prime_factorization(n):
    """Given a number, n, returns a map of n's prime factors and their frequency"""
    if is_prime(n):
        return {n : 1}

    factor_1, factor_2 = factor(n)

    factor_1_factors = prime_factorization(factor_1)
    factor_2_factors = prime_factorization(factor_2)

    aggregate_factors = factor_1_factors
    for key in factor_2_factors.keys():
        aggregate_factors[key] = factor_2_factors[key] + aggregate_factors.get(key, 0)

    return aggregate_factors

assert prime_factorization(5) == {5 : 1}
assert prime_factorization(10) == {5 : 1, 2 : 1}
assert prime_factorization(9) == {3 : 2}
assert prime_factorization(18) == {3 : 2, 2 : 1}

def least_common_multiple(x, y):
    """LCM Rules: http://www.math.com/school/subject1/lessons/S1U3L3DP.html"""
    x_primes = prime_factorization(x)
    y_primes = prime_factorization(y)

    lcm_primes = x_primes

    for key in y_primes.keys():
        lcm_primes[key] = max(y_primes[key], lcm_primes.get(key, -1))

    multiple = 1

    for key in lcm_primes.keys():
        multiple *= (key ** lcm_primes[key])

    return multiple

assert least_common_multiple(5, 2) == least_common_multiple(2, 5) == 10
assert least_common_multiple(3, 6) == least_common_multiple(6, 3) == 6
assert least_common_multiple(3, 3) == 3
assert least_common_multiple(20, 19) == least_common_multiple(20, 19) == 20*19

def smallest_multiple(n):
    """Finds the least common multiple of all numbers 1 through n
    Example: smallest_multiple(10)
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    5,2 ; 3,3 ; 2,2,2 ; 7 ; 3,2 ; 5 ; 2,2 ; 3 ; 2 ; 1
    {1, 2,2,2, 3,3, 5, 7} = 2520
    """
    multiple_so_far = n
    for i in range(n-1, 1, -1):
        multiple_so_far = least_common_multiple(x=multiple_so_far, y=i)

    return multiple_so_far

assert smallest_multiple(10) == 2520
smallest_multiple(20)

################################################################################
# 6
################################################################################

def sum_of_squares(n):
    """Returns the sum of the squares of the first n numbers"""
    return sum([i**2 for i in range(n+1)])

assert sum_of_squares(10) == 385

def square_of_sum(n):
    """Returns the square of the sum of the first n numbers"""
    return sum([i for i in range(n+1)])**2

assert square_of_sum(10) == 3025

def sum_square_difference(n):
    """Returns the difference between the sum of squares of the first n numbers and square of sum of the first n numbers"""
    return abs(sum_of_squares(n) - square_of_sum(n))

assert sum_square_difference(10) == 2640
sum_square_difference(100)

################################################################################
# 7
################################################################################

def nth_prime(n):
    """Returns the nth prime, given that 2 is the 1st and 13 is the 6th"""
    primes = []

    index = 2
    while len(primes) < n:
        if is_prime(index):
            primes.append(index)

        index += 1

    return primes[n-1]

assert nth_prime(1) == 2
assert nth_prime(6) == 13
nth_prime(10001)

################################################################################
# 8
################################################################################

def slices_of_series(series, slice_length):
    """Returns a list of iterative slices from series of slice_length"""
    return [series[index:index+slice_length] for index in range(len(series)) if len(series[index:index+slice_length]) == slice_length]

assert set(slices_of_series(series="123", slice_length=1)) == set(slices_of_series(series="321", slice_length=1)) == set(["1", "2", "3"])
assert set(slices_of_series(series="123", slice_length=2)) == set(["12", "23"])
assert set(slices_of_series(series="321", slice_length=2)) == set(["21", "32"])
assert set(slices_of_series(series="123", slice_length=3)) == set(["123"])

def string_slice_product(string_slice):
    """Returns the product of the individual single-digit numbers contained within the string_slice"""
    numbered_list = [int(string_number) for string_number in list(string_slice)]
    product = 1
    for number in numbered_list:
        product *= number
    return product

assert string_slice_product("123") == string_slice_product("321") == 1*2*3
assert string_slice_product("999") == 9**3
assert string_slice_product("9999999999999999999999999999999990") == 0


def largest_product_in_a_series(series, num_adjacent_digits):
    """Returns the largest product of a given number of adjacent digits in a series of digits"""
    assert type(series) == str and type(num_adjacent_digits) == int and (0 < num_adjacent_digits <= len(series))
    possible_largest_sequences = slices_of_series(series=series, slice_length=num_adjacent_digits)
    possible_largest_sequences_trim_zeroes = [sequence for sequence in possible_largest_sequences if "0" not in sequence]
    largest_product = max([string_slice_product(string_slice) for string_slice in possible_largest_sequences_trim_zeroes])
    return largest_product

# Loading large data string
with open("data/thousand_digit_series.txt", "r") as series_file:
    thousand_digit_series = series_file.read().replace("\n", "")

assert largest_product_in_a_series(series=thousand_digit_series, num_adjacent_digits=4) == 9*9*8*9
largest_product_in_a_series(series=thousand_digit_series, num_adjacent_digits=13)

################################################################################
# 9
################################################################################
def triplet_compute(a, b, desired_abc_sum):
    """Given a and b, finds the pythagorean triple (a, b, c) and returns a*b*c if a+b+c = desired_abc_sum or -1 otherwise"""
    c = (a**2 + b**2)**.5
    if not c.is_integer():
        return -1

    c = int(c)

    abc_sum = a + b + c
    if abc_sum != desired_abc_sum:
        return -1

    abc_product = a * b * c
    return abc_product

assert triplet_compute(a=3, b=4, desired_abc_sum=3+4+5) == 3*4*5
assert triplet_compute(a=3, b=4, desired_abc_sum=3+4+6) == -1

def special_pythagorean_triplet(n):
    """Given a number n, finds the pythagorean triplet whose sum is equal to n and returns the product of the pythagorean triplet. This function handles the iteration of a and b. triplet_compute determines if a and b are part of the correct triplet"""

    for a in range(n//2, 1, -1):
        for b in range(a, 1, -1):
            triplet_product = triplet_compute(a=a, b=b, desired_abc_sum=n)
            if triplet_product is not -1:
                return triplet_product

    return "No triplet found"

assert special_pythagorean_triplet(3+4+5) == 3*4*5
assert special_pythagorean_triplet(3+4+6) == "No triplet found"
assert special_pythagorean_triplet(5+12+13) == 5*12*13

special_pythagorean_triplet(1000)

################################################################################
# 10
################################################################################

def sum_of_primes(n):
    """Finds the sum of all primes below n"""
    return sum([i for i in range(2, n) if is_prime(i)])

assert sum_of_primes(3) == 2
assert sum_of_primes(10) == 2 + 3 + 5 + 7 == 17

sum_of_primes(2000000)

################################################################################
# 11
################################################################################
def find_sequence(grid, seq_length, x_start, y_start, x_direction, y_direction):
    """ Given a grid, a coordinate to begin, a 'direction' to move, and a 'length' to move for, finds a sequence at most as long as seq_length"""
    assert type(grid) is list and len(grid) > 0
    assert type(seq_length) == type(x_start) == type(y_start) == type(x_direction) == type(y_direction)
    assert seq_length >= 0 and -1 <= x_direction <= 1 and -1 <= y_direction <= 1

    if seq_length == 0 or x_start < 0 or x_start >= len(grid) or y_start < 0 or y_start >= len(grid[x_start]):
        return []

    current_element_of_seq = grid[x_start][y_start]

    return [current_element_of_seq] + find_sequence(grid=grid, seq_length=seq_length-1, x_start=x_start+x_direction, y_start=y_start+y_direction, x_direction=x_direction, y_direction=y_direction)

test_grid = [[0.0, 0.1], [1.0, 1.1]]
test_calls = [
    # Valid calls from (0, 0)
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=1, y_direction=0),
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=0, y_direction=1),
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=1, y_direction=1),
    # Invalid calls from (0, 0)
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=-1, y_direction=0),
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=0, y_direction=-1),
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=-1, y_direction=-1),
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=0, x_direction=1, y_direction=-1),
    # Valid calls from (1, 1)
    find_sequence(grid=test_grid, seq_length=2, x_start=1, y_start=1, x_direction=-1, y_direction=0),
    find_sequence(grid=test_grid, seq_length=2, x_start=1, y_start=1, x_direction=0, y_direction=-1),
    find_sequence(grid=test_grid, seq_length=2, x_start=1, y_start=1, x_direction=-1, y_direction=-1),
    # Valid calls from (0, 1) and (1, 0)
    find_sequence(grid=test_grid, seq_length=2, x_start=1, y_start=0, x_direction=-1, y_direction=1),
    find_sequence(grid=test_grid, seq_length=2, x_start=0, y_start=1, x_direction=1, y_direction=-1)
]
expected_results = [
    [0.0, 1.0], [0.0, 0.1], [0.0, 1.1],
    [0.0], [0.0], [0.0], [0.0],
    [1.1, 0.1], [1.1, 1.0], [1.1, 0.0],
    [1.0, 0.1], [0.1, 1.0]
]
assert test_calls == expected_results

def convert_grid_string_to_2D_list(grid_string, num_rows, num_columns):
    """Given a string of space-seperated integers, converts to a 2D list grid of integers"""
    grid_1D = [int(string_num) for string_num in grid_string.split(" ") if string_num is not ""]
    grid_2D = [[0 for row_index in range(num_rows)] for column_index in range(num_columns)]

    for column_index in range(num_columns):
        for row_index in range(num_rows):
            grid_2D[column_index][row_index] = grid_1D[column_index * num_rows + row_index]
            #print(column_index * num_rows + row_index)

    return grid_2D

test_grid_string = "1 2 3 4 5 6"
test_calls = [
    convert_grid_string_to_2D_list(grid_string=test_grid_string, num_rows=1, num_columns=6),
    convert_grid_string_to_2D_list(grid_string=test_grid_string, num_rows=2, num_columns=3),
    convert_grid_string_to_2D_list(grid_string=test_grid_string, num_rows=3, num_columns=2),
    convert_grid_string_to_2D_list(grid_string=test_grid_string, num_rows=6, num_columns=1)
]
expected_results = [
    [[1], [2], [3], [4], [5], [6]],
    [[1, 2], [3, 4], [5, 6]],
    [[1, 2, 3], [4, 5, 6]],
    [[1, 2, 3, 4, 5, 6]],
]
assert test_calls == expected_results

def list_product(lst):
    """Collapses a list to the product of its elements"""
    total = 1
    for element in lst:
        total *= element
    return total

def largest_product_in_a_grid(grid_string, num_rows, num_columns, seq_length):
    grid_2D = convert_grid_string_to_2D_list(grid_string=grid_string, num_rows=num_rows, num_columns=num_columns)

    # Getting all possible sequences -- horizontal, vertical, forwardslash and backslash
    sequences = []
    for column_index in range(num_columns):
        for row_index in range(num_rows):
            for x_direction in (0, 1):
                for y_direction in (-1, 0, 1):
                    if x_direction == y_direction == 0:    # Static direction case
                        continue

                    sequences.append(find_sequence(grid=grid_2D, seq_length=seq_length, x_start=column_index, y_start=row_index, x_direction=x_direction, y_direction=y_direction))

    # Return the maximum product of each sequence
    return max([list_product(sequence) for sequence in sequences])

test_calls = [
    # seq_length=2 calls
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=1, num_columns=6, seq_length=2),
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=2, num_columns=3, seq_length=2),
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=3, num_columns=2, seq_length=2),
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=6, num_columns=1, seq_length=2),
    # seq_length=5 calls
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=1, num_columns=6, seq_length=5),
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=2, num_columns=3, seq_length=5),
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=3, num_columns=2, seq_length=5),
    largest_product_in_a_grid(grid_string=test_grid_string, num_rows=6, num_columns=1, seq_length=5)
]
expected_results = [5*6 for i in range(4)] + [2*3*4*5*6, 2*4*6, 4*5*6, 2*3*4*5*6]
assert test_calls == expected_results

# Load and clean data
with open("data/twenty_square_grid.txt", "r") as grid_file:
    twenty_square_grid = grid_file.read().replace("\n", " ")

largest_product_in_a_grid(grid_string=twenty_square_grid, num_rows=20, num_columns=20, seq_length=4)

################################################################################
# 12
################################################################################
def combinations(lst, num_elements):
    """Given a list, will return a list of tuples of every combination of num_elements long"""
    return list(itertools.combinations(lst, num_elements))

def factorization(n):
    """Given a number, n, returns a set of n's factors"""
    # Get a list of prime factors of n
    prime_factors_dict = prime_factorization(n)
    prime_factors_expanded = list(itertools.chain.from_iterable(([[key]*value for (key, value) in prime_factors_dict.items()])))

    # Find all combinations (and products) of prime factors
    factor_combinations = []
    for num_elements in range(1, len(prime_factors_expanded)+1):
        factor_combinations += combinations(lst=prime_factors_expanded, num_elements=num_elements)

    factors = set([list_product(factor_permutation) for factor_permutation in factor_combinations] + [1, n])

    return factors

assert factorization(2) == set([1, 2])
assert factorization(3) == set([1, 3])
assert factorization(10) == set([1, 2, 5, 10])
assert factorization(12) == set([1, 2, 3, 4, 6, 12])

def highly_divisible_triangular_number(num_divisors):
    max_num_divisors_found = 0
    triangular_numbers = [0]
    index = 1

    while max_num_divisors_found < num_divisors:
        # Calculate the next triangular number
        next_triangular_number = triangular_numbers[index - 1] + index

        # Find its number of factors
        factors = factorization(next_triangular_number)
        max_num_divisors_found = max(max_num_divisors_found, len(factors))

        triangular_numbers.append(next_triangular_number)
        index += 1

    return triangular_numbers[-1]

assert highly_divisible_triangular_number(num_divisors=5) == 28
highly_divisible_triangular_number(num_divisors=500)

################################################################################
# 13
################################################################################

def sum_large_numbers(lst, first_x_digits):
    """Given a list of strings of large numbers, finds the first x digits of the sum"""
    total = 0
    for string_num in lst:
        total += int(string_num)

    return str(total)[:first_x_digits]

# Load and clean data
with open("data/fifty_digit_numbers.txt", "r") as fifty_digit_numbers_file:
    fifty_digit_numbers = fifty_digit_numbers_file.read().split()

sum_large_numbers(lst=fifty_digit_numbers, first_x_digits=10)

################################################################################
# 14
################################################################################
def collatz_sequence(n, sequence=None, memoize_dict=None):
    """Given a number n, and a listed sequence up to that number n, returns a list of the collatz sequence of that number"""
    if sequence is None:
        sequence = list()

    if (memoize_dict is not None) and (n in memoize_dict):
        return sequence + memoize_dict[n]

    sequence.append(n)

    if n == 1:
        rest_of_sequence = []
    elif n % 2 == 0:
        rest_of_sequence = collatz_sequence(n=n//2, memoize_dict=memoize_dict)
    else:
        rest_of_sequence = collatz_sequence(n=3*n+1, memoize_dict=memoize_dict)

    full_sequence = sequence + rest_of_sequence
    if (memoize_dict is not None):
        memoize_dict[n] = full_sequence

    return full_sequence


assert collatz_sequence(1) == [1]
assert collatz_sequence(2) == [2, 1]
assert collatz_sequence(3) == [3, 10, 5, 16, 8, 4, 2, 1]

memoize_dict = dict()
assert collatz_sequence(1, memoize_dict=memoize_dict) == [1] and len(memoize_dict) == 1
assert collatz_sequence(2, memoize_dict=memoize_dict) == [2, 1] and len(memoize_dict) == 2
assert collatz_sequence(3, memoize_dict=memoize_dict) == [3, 10, 5, 16, 8, 4, 2, 1] and len(memoize_dict) == 8

def longest_collatz_sequence(n):
    """Given a number n, returns a tuple of (the number between 1 and n that has the longest collatz sequence, and the length of the sequence)"""
    memoize_dict = dict()
    sequence_lengths = [len(collatz_sequence(n=i, memoize_dict=memoize_dict)) for i in range(1, n)]

    return np.argmax(sequence_lengths) + 1, np.max(sequence_lengths)

assert longest_collatz_sequence(3+1) == (3, len(collatz_sequence(3)))

longest_collatz_sequence(1000000)

################################################################################
# 15
################################################################################
def generate_grid(grid_dimension):
    """Given a grid dimension, returns a square 2D list (list of lists) with a dimension of grid_dimension x grid_dimension filled with zeroes"""
    return [[0 for i in range(grid_dimension)] for j in range(grid_dimension)]

assert generate_grid(1) == [[0]]
assert generate_grid(2) == [[0, 0], [0, 0]]
assert generate_grid(3) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def prior_positions(x, y, grid_dimension):
    """Given an (x,y) coordinate and the grid dimension returns a list of (x,y) positions that can lead to the given position, assuming only right and down moves are allowed"""
    priors = []

    if x > 0:
        priors.append((x-1, y))

    if y > 0:
        priors.append((x, y-1))

    return priors

assert prior_positions(0, 0, grid_dimension=20) == []
assert prior_positions(0, 1, grid_dimension=20) == prior_positions(1, 0, grid_dimension=20) == [(0,0)]
assert prior_positions(1, 1, grid_dimension=20) == [(0,1), (1,0)]

def set_grid_value(value, x, y, grid):
    """Abstraction for setting the values of an (x,y) coordinate on the grid"""
    assert 0 <= x < len(grid)
    assert 0 <= y < len(grid)
    grid[x][y] = value

def get_grid_value(x, y, grid):
    """Abstraction for returning the value of an (x,y) coordinate on the grid"""
    assert 0 <= x < len(grid)
    assert 0 <= y < len(grid[x])
    return grid[x][y]

def lattice_paths_grid(grid_dimension):
    """Given a grid dimension, determines the number of possible routes from the top left corner to the bottom right corner only being able to move right and down

    Grid Approach
    Note: From any position on the grid, the number of paths to that position is the sum of the number of paths of the two positions that can lead to that position
    """
    # Example given in problem description walks along *edges* so this must be simulated with a grid of with dimension+1
    real_grid_dimension = grid_dimension + 1
    grid = generate_grid(real_grid_dimension)

    # Starting Path
    set_grid_value(value=1, x=0, y=0, grid=grid)

    for x in range(real_grid_dimension):
        for y in range(real_grid_dimension):
            priors = prior_positions(x, y, real_grid_dimension)

            total_paths = 0
            for prior in priors:
                total_paths += get_grid_value(x=prior[0], y=prior[1], grid=grid)

            if total_paths != 0:
                set_grid_value(value=total_paths, x=x, y=y, grid=grid)

    return int(get_grid_value(x=grid_dimension, y=grid_dimension, grid=grid))

def lattice_paths_combinatorics(grid_dimension):
    """Given a grid dimension, determines the number of possible routes from the top left corner to the bottom right corner only being able to move right and down

    Combinatorics Approach
    Note: At any point, there are two possible choices, except in the special cases in which you are at the rightmost or downmost edge
    Note: These special cases are only possible AFTER grid_dimension moves have been made
    Note: The total number of moves that can be made sum to grid_dimension * 2
    Reframe: This is an identical situation to there being a deck of 20 moves or cards, 10 A's and 10 B's, and you sampling 20 times from that deck without replacement.
    What is the total number of possible orderings? => 20 choose 10
    """
    from scipy.special import comb
    return int(comb(N=int(grid_dimension*2), k=grid_dimension))

assert lattice_paths_grid(2) == lattice_paths_combinatorics(2) == 6
assert lattice_paths_grid(20) == lattice_paths_combinatorics(20)

lattice_paths_grid(20)

################################################################################
# 16
################################################################################

def power_digit_sum(base, exponent):
    """Returns the sum of the digits of the number that is equal to the base to the power of the exponent"""
    return sum([int(digit) for digit in str(base ** exponent)])

assert power_digit_sum(base=2, exponent=3) == 2**3
assert power_digit_sum(base=2, exponent=4) == 1+6
assert power_digit_sum(base=10, exponent=2) == 1

power_digit_sum(base=2, exponent=1000)

################################################################################
# 17
################################################################################

def number_letter_counts(start_num, end_num):
    """Returns the number of letters used to write out numbers start_num through end_num in words, inclusive"""
    assert type(start_num) == type(end_num) == int
    from num2words import num2words

    nums = range(start_num, end_num+1)
    words = [num2words(num) for num in nums]

    return len("".join(words).replace("-", "").replace(" ", ""))

assert number_letter_counts(start_num=1, end_num=1) == len("one")
assert number_letter_counts(start_num=1, end_num=2) == len("one"+"two")
assert number_letter_counts(start_num=1, end_num=500) == number_letter_counts(start_num=1, end_num=499) + len("fivehundred")


number_letter_counts(start_num=1, end_num=1000)

################################################################################
# 18
################################################################################

def generate_triangle(triangle_string):
    """Given a triangle string with \n seperated rows and \t seperated columns, generates and returns a 2D list of integers representing the triangle"""
    rows = triangle_string.split("\n")
    return [[int(num) for num in row.split()] for row in rows if len(row) > 0]

test_triangle_string = "1\n2\t3\n4\t5\t6"
test_triangle_grid = generate_triangle(test_triangle_string)
assert test_triangle_grid == [[1], [2, 3], [4, 5, 6]]

def get_next_possible_coordinates(grid, x, y, x_reach=[0, 1], y_reach=[1]):
    """Returns a list of (x,y) coordinates that are reachable from (x,y) on grid given the reach constraints. Conceptually defaults to 'down' and 'directly adjacent' from the given (x,y) coordinate on the grid. Returns an empty list if no other coordinates are reachable"""
    assert type(x) == type(y) == int
    assert type(x_reach) == type(y_reach) == type(grid) == list
    assert 0 <= y < len(grid) and 0 <= x < len(grid[y])

    next_possible_coordinates = []

    for x_change in x_reach:
        for y_change in y_reach:
            next_possible_coordinate_x = x + x_change
            next_possible_coordinate_y = y + y_change
            if not (0 <= next_possible_coordinate_y < len(grid)):
                continue
            elif not (0 <= next_possible_coordinate_x < len(grid[next_possible_coordinate_y])):
                break
            else:
                next_possible_coordinates.append((next_possible_coordinate_x, next_possible_coordinate_y))

    return next_possible_coordinates

assert get_next_possible_coordinates(grid=test_triangle_grid, x=0, y=0) == [(0, 1), (1, 1)]
assert get_next_possible_coordinates(grid=test_triangle_grid, x=0, y=2) == []
assert get_next_possible_coordinates(grid=test_triangle_grid, x=2, y=2) == []

def maximum_path_sum(triangle_string):
    """Given a triangle string, with \n seperated rows and \t seperated columns, returns a tuple containing the maximum path sum and a list of the order of the maximum path)"""
    triangle_grid = generate_triangle(triangle_string)
    num_rows = len(triangle_grid)
    top_of_triangle_grid = (0, 0)

    # Populate memoize_dict with each coordinate
    # Memoize dict stores (k=coordinate, v=[most_optimal_next_coordinate, sum_of_most_optimal_path_from_this_coordinate]) pairs
    memoize_dict = {}
    for row_index in range(num_rows-1, 0-1, -1):
        row = triangle_grid[row_index]
        for column_index in range(len(row)):
            # Generalized grid access using (x,y) is conceptually reversed with triangle grids
            current_coordinate = (column_index, row_index)
            current_coordinate_value = get_grid_value(x=row_index, y=column_index, grid=triangle_grid)

            next_possible_coordinates = get_next_possible_coordinates(grid=triangle_grid, x=column_index, y=row_index)

            optimal_next_coordinate = None
            optimal_next_value = 0

            for next_possible_coordinate in next_possible_coordinates:

                if next_possible_coordinate in memoize_dict:
                    possible_optimal_next_value = memoize_dict[next_possible_coordinate][1]
                else:
                    # Generalized grid access using (x,y) is conceptually reversed with triangle grids
                    possible_optimal_next_value = get_grid_value(x=next_possible_coordinate[1], y=next_possible_coordinate[0])

                if optimal_next_value < possible_optimal_next_value:
                    optimal_next_value = possible_optimal_next_value
                    optimal_next_coordinate = next_possible_coordinate

            # Memoize what has been determined to be the optimal next coordinate and the maximum path sum onwards from the current coordinate
            memoize_dict[current_coordinate] = (optimal_next_coordinate, current_coordinate_value + optimal_next_value)

    # Compute the optimal path and its sum starting from the top of the triangle using the memoize_dict
    next_coordinate_of_optimal_path, optimal_path_sum = memoize_dict[top_of_triangle_grid]
    # Generalized grid access using (x,y) is conceptually reversed with triangle grids
    optimal_path = [get_grid_value(x=top_of_triangle_grid[1], y=top_of_triangle_grid[0], grid=triangle_grid)]
    while next_coordinate_of_optimal_path is not None:
        optimal_path.append(get_grid_value(x=next_coordinate_of_optimal_path[1], y=next_coordinate_of_optimal_path[0], grid=triangle_grid))
        next_coordinate_of_optimal_path = memoize_dict[next_coordinate_of_optimal_path][0]

    return (optimal_path, optimal_path_sum)

# Loading large data strings
with open("data/four_row_triangle.txt", "r") as triangle_file:
    four_row_triangle_string = triangle_file.read()

with open("data/fifteen_row_triangle.txt", "r") as triangle_file:
    fifteen_row_triangle_string = triangle_file.read()

assert maximum_path_sum(four_row_triangle_string) == ([3, 7, 4, 9], 23)
maximum_path_sum(fifteen_row_triangle_string)

################################################################################
# 19
################################################################################

def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

assert is_leap_year(4) == True
assert is_leap_year(5) == False
assert is_leap_year(500) == False
assert is_leap_year(800) == True

def get_num_days_in_month(month: int, year: int) -> int:
    """@param month : an integer between 1 and 12, mapping to the month of the year
       @param year : the year in which the month occurs
       @return : Returns an integer, the the number of days that occur in that month
    """
    month_days = {
        1 : 31,
        2 : 28,
        3 : 31,
        4 : 30,
        5 : 31,
        6 : 30,
        7 : 31,
        8 : 31,
        9 : 30,
        10 : 31,
        11 : 30,
        12 : 31
    }

    if is_leap_year(year):
        month_days[2] += 1

    return month_days[month]

def advance_to_next_month(reference_day: int, reference_date: date, current_date: date) -> Tuple[int, date]:
    """@param reference_day: an integer between 0 and 6, mapping to a particular day of the week, which the problem offers as a reference
       @param reference_date: the date, which the problem offers as a reference -- must be the start of the month
       @param current_date: the current date -- must be the start of the month
       @return: a tuple containing a new current_day and a new
    """
    assert 0 <= reference_day <= 6
    assert reference_date <= current_date
    assert current_date.day == 1

    new_current_date = current_date + timedelta(days=get_num_days_in_month(month = current_date.month, year = current_date.year))
    days_offset = (new_current_date - reference_date).days
    new_current_day = (reference_day + days_offset ) % 7

    return (new_current_day, new_current_date)

assert advance_to_next_month(reference_day = 1, reference_date = date(year = 1900, month = 1, day = 1), current_date=date(year = 2019, month = 9, day = 1)) == (2, date(year = 2019, month = 10, day = 1))

assert advance_to_next_month(reference_day = 1, reference_date = date(year = 1900, month = 1, day = 1), current_date=date(year = 2019, month = 10, day = 1)) == (5, date(year = 2019, month = 11, day = 1))

def is_sunday(day: int, sunday: int = 0) -> bool:
    """@param day: an integer between 0 and 6, mapping to a particular day of the week
       @return: a boolean indicating whether the given day is a sunday or not
    """
    return day == sunday

def counting_sundays(reference_day: int, reference_date: date, start_date: date, end_date: date) -> int:
    """@param reference_day: an integer between 0 and 6, mapping to a particular day of the week, which the problem offers as a reference
       @param reference_date: the date, which the problem offers as a reference -- must be the start of the month
       @param start_date: the date when counting should begin, inclusive
       @param end_date: the date when counting should conclude, inclusive
       @note day_map: days = {
           0 : "Sunday",
           1 : "Monday",
           2 : "Tuesday",
           3 : "Wednesday",
           4 : "Thursday",
           5 : "Friday",
           6 : "Saturday"
       }
       @return: the number of Sundays that fall on the first day of a month
    """
    assert 0 <= reference_day <= 6
    assert reference_date <= start_date <= end_date
    assert reference_date.day == 1

    # Define reference position
    current_date: date = reference_date
    current_day: int = reference_day
    total_sundays: int = 0
    pre_start_sundays: int = 0

    # Count all Sundays between start_date and end_date
    while current_date < end_date:
        assert current_date.day == 1
        total_sundays += int(is_sunday(day = current_day)) * int(start_date <= current_date <= end_date)

        # Move to next month
        current_day, current_date = advance_to_next_month(
            reference_day = reference_day,
            reference_date = reference_date,
            current_date = current_date
        )

    return total_sundays

counting_sundays(reference_day = 1, reference_date = date(year = 1900, month = 1, day = 1), start_date = date(year = 1901, month = 1, day = 1), end_date = date(year = 2000, month = 12, day = 31))

################################################################################
# 20
################################################################################

def string_slice_sum(string_slice: str) -> int:
    """Returns the sum of the individual single-digit numbers contained within the string_slice"""
    return sum([int(character) for character in string_slice])

assert string_slice_sum(string_slice = "123") == 1 + 2 + 3
assert string_slice_sum(string_slice = "000") == 0

def factorial(n: int) -> int:
    """@param n: an integer, from which to generate n!
       @return: an integer, representing the factorial of n
    """
    # if n == 0:
    #     return 1
    return reduce(lambda x,y: x*y, [i for i in range(1, n+1)], 1)

assert factorial(n = 0) == 1
assert factorial(n = 1) == 1
assert factorial(n = 2) == 2*1
assert factorial(n = 100) == factorial(99) * 100

def factorial_digit_sum(n: int) -> int:
    """@param n: an integer, from which to generate n! and determine the sum of digits of
       @return: an integer, representing the digit sum of the factorial of n
    """
    return string_slice_sum(string_slice = str(factorial(n = n)))

factorial_digit_sum(n = 100)

################################################################################
# 21
################################################################################

def proper_divisors(n: int) -> Set:
    """Returns all the proper divisors of n (excluding n itself)"""
    divisors = factorization(n = n)
    divisors.remove(n)
    return divisors

def sum_amicable_numbers(n: int) -> int:
    """Returns the sum of all amicable numbers less than the integer, n"""
    factors = {i : proper_divisors(i) for i in range(2, n+1)}
    proper_divisor_sum = {i : (sum(factors[i])) for i in factors.keys()}
    assert proper_divisor_sum[220] == 284 and proper_divisor_sum[284] == 220
    amicable_numbers = {i : proper_divisor_sum.get(proper_divisor_sum[i]) == i and proper_divisor_sum[i] != i for i in proper_divisor_sum.keys()}
    assert amicable_numbers[220] == True and amicable_numbers[284] == True
    return sum([i for i in amicable_numbers.keys() if amicable_numbers[i] == True])

sum_amicable_numbers(n = 10000)

################################################################################
# 22
################################################################################

def get_letter_score(letter: str) -> int:
    """Returns the alphabetical score of a letter. Example: 'A' is 1 and 'C' is 3"""
    assert type(letter) == str and len(letter) == 1
    letter = letter.lower()
    letter_string = "abcdefghijklmnopqrstuvwxyz"
    assert letter in letter_string
    letters = [letter for letter in letter_string]
    letter_scores = {letter : score for score, letter in enumerate(letters, start=1)}
    return letter_scores[letter]

assert get_letter_score(letter = 'a') == 1
assert get_letter_score(letter = 'z') == 26

def get_name_score(name: str) -> int:
    """Returns the alphabetical score of a name. Example: 'COLIN' is 3 + 15 + 12 + 9 + 14 = 53"""
    return sum([get_letter_score(letter = letter) for letter in name])

assert get_name_score(name = "COLIN") == 53

def names_scores():
    """Returns the sum of the name scores in the names.txt, where a name score is defined as the multiple of the alphabetically sorted position """
    # Load and clean data
    with open("data/names.txt", "r") as names_file:
        names = sorted(names_file.read().replace('"', '').lower().split(","))

    return sum([index * get_name_score(name = name) for index, name in enumerate(names, start=1)])

names_scores()

################################################################################
# 23
################################################################################

def is_abundant(n: int) -> bool:
    return sum(proper_divisors(n)) > n

assert is_abundant(12)

def get_abundant_numbers(n: int = 28123) -> Set:
    """Returns a set of all abundant numbers less than n"""
    return {i for i in range(12, n) if is_abundant(i)}

def all_sums_of_two_abundants(abundant_numbers: Set) -> Set:
    """Returns a set of all possible sums from two abundant numbers"""
    all_sums = set()
    for i in abundant_numbers:
        for j in abundant_numbers:
            all_sums.add(i+j)
    return all_sums

def non_abundant_sums() -> int:
    """Returns the sum of all the positive integers which cannot be written as the sum of two abundant numbers"""
    n = 28123
    abundant_numbers = get_abundant_numbers(n)
    abundant_number_sums = all_sums_of_two_abundants(abundant_numbers)
    assert n+1 in abundant_number_sums
    return sum([i for i in range(n) if i not in abundant_number_sums])

non_abundant_sums()

################################################################################
# 24
################################################################################
