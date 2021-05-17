import os


def get_data_path(data_name, os_type):
    data_path = ""
    working_dir = os.getcwd()
    if os_type == "windows":
        if "tests" in working_dir:
            working_dir_list = working_dir.split('\\')
            working_dir_list = working_dir_list[:-1]
            working_dir = '\\'.join(working_dir_list)
        data_path = working_dir + '\\data\\' + data_name
    else:
        if "tests" in working_dir:
            working_dir_list = working_dir.split("/")
            working_dir_list = working_dir_list[:-1]
            working_dir = "/".join(working_dir_list)
        data_path = working_dir + '/data/' + data_name
    return data_path


if __name__ == "__main__":
    # os test
    data_name = 'sonar.csv'
    example_windows_path = 'C:\\Users\\kerem\\Masaüstü\\artificial_neural_network_automation\\tests'
    print(example_windows_path)
    working_dir_list = example_windows_path.split('\\')
    working_dir_list = working_dir_list[:-1]
    example_windows_path = '\\'.join(working_dir_list)
    data_path = example_windows_path + '\\data\\' + data_name
    print(data_path)
