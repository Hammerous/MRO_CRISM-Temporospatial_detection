def convert_pid(str):
    str = str.split("_")
    str[-2] = str[-2].replace("if", "sr")
    return "_".join(str)

def reverse_pid(str):
    str = str.split("_")
    str[-2] = str[-2].replace("sr", "if")
    return "_".join(str)