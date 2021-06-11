import time


def execution_time(time_type):
    def compute_execution_time(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            value = func(*args, **kwargs)
            end_time = time.time()
            exe_time_seconds = end_time - start_time
            exe_time_minutes = exe_time_seconds / 60
            exe_time_hours = exe_time_minutes / 60
            if time_type == "seconds":
                print("Method Execution Time : {:.2f}".format(exe_time_seconds) + " seconds")
            elif time_type == "minutes":
                print("Method Execution Time : {:.2f}".format(exe_time_minutes) + " minutes")
            elif time_type == "hours":
                print("Method Execution Time : {:.2f}".format(exe_time_hours) + " hours")
            return value
        return wrapper
    return compute_execution_time
