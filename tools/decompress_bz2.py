#!/usr/bin/env python3
"""Decompress a .bz2 CSV into an uncompressed CSV file.
Usage: python tools/decompress_bz2.py resources/large_flights_500k.csv.bz2 resources/large_flights_500k.csv
"""
import bz2
import sys
import shutil

def decompress(in_path, out_path):
    with bz2.open(in_path, 'rb') as inf, open(out_path, 'wb') as outf:
        shutil.copyfileobj(inf, outf)

def main():
    if len(sys.argv) < 3:
        print('Usage: decompress_bz2.py <in.bz2> <out.csv>')
        sys.exit(2)
    decompress(sys.argv[1], sys.argv[2])
    print('Decompressed to', sys.argv[2])

if __name__ == '__main__':
    main()
