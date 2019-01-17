################################################################################
# Project Euler Practice
# Author: kmishra9
# Previous Attempts: https://drive.google.com/file/d/0BwPzCEOJw0myMTl2STJaQjBoOGs/view
################################################################################

import os
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
with open("data/thousand_digit_series.txt", 'r') as series_file:
    thousand_digit_series = series_file.read().replace('\n', '')

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
