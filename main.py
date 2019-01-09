################################################################################
# Project Euler Practice
# Author: kmishra9
################################################################################

################################################################################
# 1
################################################################################

def multiples_of_3_or_5(n):
    return sum([i for i in range(n) if i % 3 == 0 or i % 5 == 0])

assert multiples_of_3_or_5(10) == 23
print(multiples_of_3_or_5(1000))

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
print(even_fibonacci_numbers())

################################################################################
# 3
################################################################################

def is_factor(n, potential_factor):
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
print(largest_prime_factor(600851475143))

################################################################################
# 4
################################################################################

def is_palindrome(n):
    assert type(n) is int
    return n == int(str(n)[::-1])

def largest_palindrome_product(top=999, bottom=100):
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
print(largest_palindrome_product())

################################################################################
# 5
################################################################################
