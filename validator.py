"""
    usage: python3 validator.py path/to/input path/to/vertex/output
"""

import sys

input_file = open(sys.argv[1], "r")

sol_file = open(sys.argv[2], "r")

n = int(input_file.readline().strip())
strings = [l.strip() for l in input_file.readlines()[1:]]

s = sol_file.readline().strip()

if len(s) != len(strings[0]):
    print("Lengths of strings do not match")
    quit()

dist = max([sum(c1 != c2 for c1, c2 in zip(s, string)) for string in strings])

print("String has max distance {}".format(dist))
