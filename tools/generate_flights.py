#!/usr/bin/env python3
"""Generate a large synthetic flights CSV with a schema similar to the 2007 dataset.
Produces a bzip2-compressed CSV at resources/large_flights_N.csv.bz2
"""
import csv
import random
import os
import argparse
import bz2
from datetime import datetime, timedelta

AIRPORTS = [
    'ATL','LAX','ORD','DFW','DEN','JFK','SFO','LAS','SEA','MCO','CLT','EWR','PHX','IAH','MIA'
]

CARRIERS = ['AA','DL','UA','WN','US','MQ','B6','AS']

def rand_time():
    h = random.randint(0,23)
    m = random.randint(0,59)
    return f"{h:02d}{m:02d}"

def gen_row(year=2007, tailnum_source=None, match_prob=0.0):
    month = random.randint(1,12)
    day = random.randint(1,28)
    dow = datetime(year, month, day).isoweekday()
    dep_time = rand_time()
    crs_dep = dep_time
    arr_time = rand_time()
    crs_arr = arr_time
    unique_carrier = random.choice(CARRIERS)
    flight_num = random.randint(1,9999)
    # choose tailnum: with probability match_prob pick from tailnum_source (if provided)
    if tailnum_source and match_prob > 0.0 and random.random() < match_prob:
        tailnum = random.choice(tailnum_source)
    else:
        tailnum = 'N' + str(random.randint(1000,99999))
    actual_elapsed = random.randint(30,300)
    crs_elapsed = actual_elapsed
    airtime = max(10, actual_elapsed - random.randint(0,20))
    arr_delay = random.randint(-30,120)
    dep_delay = max(-20, arr_delay - random.randint(-5,10))
    origin = random.choice(AIRPORTS)
    dest = random.choice([a for a in AIRPORTS if a != origin])
    distance = random.randint(50,2500)
    taxi_in = random.randint(0,20)
    taxi_out = random.randint(0,30)
    cancelled = 0
    diverted = 0
    carrier_delay = 0
    weather_delay = 0
    nas_delay = 0
    security_delay = 0
    late_aircraft_delay = 0

    return [year, month, day, dow, dep_time, crs_dep, arr_time, crs_arr, unique_carrier,
            flight_num, tailnum, actual_elapsed, crs_elapsed, airtime, arr_delay, dep_delay,
            origin, dest, distance, taxi_in, taxi_out, cancelled, '', diverted,
            carrier_delay, weather_delay, nas_delay, security_delay, late_aircraft_delay]

def generate(path, rows=200000, match_prob=None):
    # Load tailnums from plane-data if available
    tailnum_source = None
    plane_data_path = os.path.join('src', 'main', 'dataset', 'plane-data.csv')
    if os.path.isfile(plane_data_path):
        try:
            with open(plane_data_path, 'r', encoding='utf-8', errors='ignore') as pf:
                tailnum_source = [line.split(',')[0].strip() for line in pf if line.strip()]
                tailnum_source = [t for t in tailnum_source if t and not t.lower().startswith('tailnum')]
        except Exception:
            tailnum_source = None

    # default match probability when plane-data is present
    if match_prob is None:
        match_prob = 0.10 if tailnum_source else 0.0

    header = [
        'Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime',
        'UniqueCarrier','FlightNum','TailNum','ActualElapsedTime','CRSElapsedTime','AirTime',
        'ArrDelay','DepDelay','Origin','Dest','Distance','TaxiIn','TaxiOut','Cancelled',
        'CancellationCode','Diverted','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay'
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # write compressed bzip2
    with bz2.open(path, 'wt') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(rows):
            writer.writerow(gen_row(tailnum_source=tailnum_source, match_prob=match_prob))
            if (i+1) % 10000 == 0:
                print(f"Generated {i+1} rows")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=200000, help='Number of rows to generate')
    parser.add_argument('--out', type=str, default='resources/large_flights_200k.csv.bz2')
    parser.add_argument('--match-prob', type=float, default=None, help='Fraction of rows whose tailnum is sampled from plane-data (0-1). If omitted, a default is used when plane-data is present')
    args = parser.parse_args()
    print(f"Generating {args.rows} rows to {args.out} ...")
    generate(args.out, rows=args.rows, match_prob=args.match_prob)
    print('Done')

if __name__ == '__main__':
    main()
