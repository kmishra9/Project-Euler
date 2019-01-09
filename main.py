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
