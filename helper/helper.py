
def contol_instance_type(object, object_name, type):
    """Contols instance type of given object.

    Args:
      object: Python object to be controlled.
      object_name: str. Name of the object.
      type: Data type of object like str, dict, int etc.
    """
    if isinstance(object, type):
        print("{} data type is valid".format(object_name))
    else:
        raise Exception("Sorry, {} cannot be anything than {}".format(object_name, str(type)))
