#!/usr/bin/python

import sys

maxx = 0
maxPath = ''

for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    if len(data_mapped) != 2:
        # Something has gone wrong. Skip this line.
        continue

    thisKey, thisCount = data_mapped

    if int(thisCount) > maxx:
        maxPath = thisKey
        maxx = int(thisCount)

print maxPath, "\t", maxx
