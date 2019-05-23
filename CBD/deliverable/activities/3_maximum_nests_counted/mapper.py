#!/usr/bin/python

# Format of each line is:

# site_name, site_id, cammlr_region, longitude, latitude, common_name, day, month, year, season_starting, penguin_count, accuracy, count_type, vantage, reference

import sys

for line in sys.stdin:
    data = line.strip().split(",")
    if len(data) == 15:
        site_name, site_id, cammlr_region, longitude, latitude, common_name, day, month, year, season_starting, penguin_count, accuracy, count_type, vantage, reference = data
        if site_id == 'site_id':
            continue 
        if count_type == 'nests':
         
            print "{0}\t{1}".format(site_name, penguin_count)
