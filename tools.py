import re

def sort_wav(filename):
    match = re.match(r'room(\d+)_scene(\d+)_mic(\d+)\.wav', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return (0, 0, 0) 


def sort_clean_wav(filename):
    match = re.match(r'signal(\d+)\.wav', filename)
    if match:
        return int(match.group(1))
    return (0)  
