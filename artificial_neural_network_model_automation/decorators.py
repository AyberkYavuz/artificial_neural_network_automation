import time


def execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        exe_time = end_time - start_time
        print("Method Execution Time : {:.2f}".format(exe_time) + " seconds")
        return value
    return wrapper
