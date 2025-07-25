#!/usr/bin/env python3
from time import time

import redis  # ty: ignore[unresolved-import]
from loguru import logger  # ty: ignore[unresolved-import]
from more_itertools import ilen

r = redis.Redis(host='localhost', port=6379, db=0)


N = 1_000_000


def items():
    yield from map(str, range(N))


TAG = 'keys'


def reset():
    r.delete(TAG)


def write():
    for i, obj in enumerate(items()):
        key = f'obj:{i}'
        r.hset(key, 'data', obj)
        r.lpush(TAG, key)


def read():
    keys = r.lrange(TAG, 0, -1)
    result = (r.hget(key, 'data') for key in keys)
    print('total', ilen(result))


# TODO could use lmove for atomic operations?
def write2():
    for obj in items():
        r.lpush(TAG, obj)


def read2():
    result = r.lrange(TAG, 0, -1)
    print('total', ilen(result))


reset()

a = time()
write2()
b = time()
logger.info(f'writing took {b - a:.1f}s')

a = time()
read2()
b = time()
logger.info(f'reading took {b - a:.1f}s')


# with read()/write()
# 100000 strings:
# 2023-09-09 01:50:23.498 | INFO     | __main__:<module>:37 - writing took 13.1s
# 2023-09-09 01:50:30.052 | INFO     | __main__:<module>:42 - reading took 6.6s
# hmm kinda slow..


# with read2/write2, writing about 7secs, and reading is instantaneous??
# for 1M objects, writing took 60 secs, and reading 0.2s?
# lol could be promising...
# I guess it's not iterative, but could retrieve items in batches?
