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
        if float(longitude) >= -74.0 and float(longitude) <= -25.0:
            print "{0}\t{1}".format('Argentina', penguin_count)
	if float(longitude) >= 142.2 and float(longitude) <= 160.0:
            print "{0}\t{1}".format('Australia', penguin_count)
        if float(longitude) >= 44.38 and float(longitude) <= 136.11:
            print "{0}\t{1}".format('Australia', penguin_count)
        if float(longitude) >= -90.0 and float(longitude) <= -53.0:
            print "{0}\t{1}".format('Chile', penguin_count)
        if float(longitude) >= 136.11 and float(longitude) <= 142.2:
            print "{0}\t{1}".format('France', penguin_count)
        if float(longitude) >= 160.0 or float(longitude) <= -150.0:
            print "{0}\t{1}".format('New Zealand', penguin_count)
        if float(longitude) >= -20.0 and float(longitude) <= 44.38:
            print "{0}\t{1}".format('Norway', penguin_count)
        if float(longitude) >= -80.0 and float(longitude) <= -20.0:
            print "{0}\t{1}".format('United Kingdom', penguin_count)
