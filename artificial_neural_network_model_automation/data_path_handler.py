import os


def get_data_path(data_name):
    data_path = ""
    working_dir = os.getcwd()
    if "tests" in working_dir:
        working_dir_list = working_dir.split("/")
        working_dir_list = working_dir_list[:-1]
        working_dir = "/".join(working_dir_list)
    data_path = working_dir + '/data/' + data_name
    return data_path
