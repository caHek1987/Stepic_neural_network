from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
print(fib(400))
print(fib.cache_info())


def cache_decorator(func):
    cache = {}
    def cache_iteration(*args, **kwargs):
        if args not in cache:
            cache[args] = func(*args, **kwargs)
        print(cache)
        return cache
    return cache_iteration


@cache_decorator
def factorial(n):
    fact = 1
    i = 1
    while i <= n:
        fact *= i
        print(i, fact)
        i += 1
    return fact

factorial(7)
factorial(6)



