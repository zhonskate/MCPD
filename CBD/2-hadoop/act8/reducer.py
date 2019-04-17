#!/usr/bin/python

import sys

salesTotal = 0
oldKey = None
maxx = 0
maxPath = ''

# Loop around the data
# It will be in the format key\tval
# Where key is the store name, val is the sale amount
#
# All the sales for a particular store will be presented,
# then the key will change and we'll be dealing with the next store

for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    if len(data_mapped) != 2:
        # Something has gone wrong. Skip this line.
        continue

    thisKey, thisSale = data_mapped

    if oldKey and oldKey != thisKey:
        #print oldKey, "\t", salesTotal
        if salesTotal > maxx:
	    maxPath = oldKey
            maxx = salesTotal
        oldKey = thisKey;

        salesTotal = 0

    oldKey = thisKey
    salesTotal += int(thisSale)

if oldKey != None:
    if salesTotal > maxx:
        maxPath = oldKey
        maxx = salesTotal

print maxPath, "\t", maxx
