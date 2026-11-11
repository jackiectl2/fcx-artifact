"""Throughput pilot for sizing the E-WIKI conversion job (no output besides timings): stream a dump through
wiki_convert.pages() and report pages, revisions, bytes read and seconds. A truncated input (e.g. `head -c`) ends with an XML
parse error, which is caught and reported.
usage: wiki_pilot.py <dump path or ->
"""
import sys, time
import xml.etree.ElementTree as ET

import wiki_convert


class Counting:
    def __init__(self, f):
        self.f, self.n = f, 0

    def read(self, k=-1):
        b = self.f.read(k)
        self.n += len(b)
        return b


def main():
    path = sys.argv[1]
    t0, pages, revs = time.time(), 0, 0
    if path == "-":
        src = Counting(sys.stdin.buffer)
        wiki_convert.opener = lambda p: __import__("contextlib").nullcontext(src)
    else:
        src = None
    try:
        for pg in wiki_convert.pages(path):
            pages += 1
            revs += len(pg["revs"])
    except ET.ParseError as e:
        print("parse stopped:", e)
    dt = time.time() - t0
    read = src.n if src else None
    print(f"PILOT pages={pages} revisions={revs} bytes={read} seconds={dt:.1f}" + (f" MB/s={read / dt / 1e6:.1f}" if read else ""), flush=True)


if __name__ == "__main__":
    main()
