#!/usr/bin/python
# -*- coding: utf-8 -*-

# Test for hashability of abstract state

from __future__ import print_function
import __builtin__
import itertools
import inspect
import collections
from subprocess import call
import subprocess
import sys
import signal
#import shelve
import hashlib
#import pickle
import atexit
import cPickle
import time

from blessed import Terminal
import functools
import numpy as np
from scipy.spatial import ConvexHull


import err
import fileops as fops


# XXX: checkout the function toolz.itertoolz.partition()
def pairwise(iterable):
    '''s -> (s0,s1), (s1,s2), (s2, s3), ...'''

    (a, b) = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def iterate_in_chunks(iterable, chunk_length):
    n = chunk_length
    for i in xrange(0, len(iterable), n):
        yield iterable[i:i + n]


def strict_call(*args, **kwargs):
    ret_val = call(*args, **kwargs)
    if ret_val != 0:
        raise err.Fatal(
            'call() returned non-zero return value: {}'.format(ret_val))


class CallError(Exception):
    pass


# subprocess.check_output
def strict_call_get_op(*args, **kwargs):
    try:
        op = subprocess.check_output(stderr=subprocess.STDOUT, *args, **kwargs)
    except subprocess.CalledProcessError as e:
        term = Terminal()

        # print term.red(e.output)
#        error_msg = str(e.output).decode('utf-8')
#        print term.red(error_msg)

        error_msg = e.output.decode('utf-8')
        print(term.red(error_msg))

        # print term.red(str(e.output))

        sys.stdout.flush()
        raise CallError(error_msg)

    return op


def call_get_op(*args, **kwargs):
    try:
        op = subprocess.check_output(stderr=subprocess.STDOUT, *args, **kwargs)
    except subprocess.CalledProcessError as e:
        term = Terminal()

        error_msg = e.output.decode('utf-8')
        print(term.red(error_msg))
        sys.stdout.flush()

    return op


class Unique(object):

    obj_dict = {}

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self, ID):

        # Check if ID is hashable

        if not isinstance(ID, collections.Hashable):
            raise err.Fatal('unhashable abstract state representation is unhandled'
                            )

        unique_obj = Unique.obj_dict.get(ID)
        if unique_obj is None:
            new_obj = self._decorated(ID)
            Unique.obj_dict[ID] = new_obj
            return new_obj
        else:
            return unique_obj

    def clear(self):
        Unique.obj_dict = {}

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


class Singleton:

    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self, *args, **kwargs):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """

        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated(*args, **kwargs)
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


def decorate(s):
    return '\n{border}\n{s}\n{border}'.format(border='=' * 20, s=str(s))


def colorize(msg, t=Terminal()):
    return t.blue(msg)


class Memodict(object):
    """ Memoization decorator for a function taking a single argument.
    Supposed to be very fast..."""
    def __init__(self, f):
        self.cache = {}
        self.f = f
        return

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return functools.partial(self, obj)

    def __call__(self, *args):
        key = args
        v = self.cache.get(key)
        if v is None:
            v = self.f(*key)
            self.cache[key] = v
        return v


def memodict(f):
    """ Memoization decorator for a function taking a single argument.
    Supposed to be very fast..."""
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


def my_fast_str(arg):
    """if the arguement is a numpy array, uses tostring instead.
    Made specially as a hash func to be used with memoization when
    numpy arrays are involved.
    """
    if isinstance(arg, collections.Iterable):
        #s = [a.tostring() if isinstance(a, np.ndarray) else str(a)
        s = [hashlib.sha1(a).hexdigest() if isinstance(a, np.ndarray) else str(a)
             for a in arg]
        return ''.join(s)
    else:
        return str(arg)


# TODO: Remove these non sense methods. Methods which have no realtion
# to the objects, should be class/static methods and this issue should
# never arrise
def memoize_hash_method(args, kwargs):
    """hash function for memoizing methods when the objects don't
    matter: ignores self!!
    """
    return my_fast_str(args[1:]) + str(kwargs)


def memoize_hash_fun(args, kwargs):
    """hash function for memoizing!!
    """
    return my_fast_str(args) + str(kwargs)


def memoize2disk(hash_fun):
    """Memoizing decorator which saves to the disk
    """
    def memoize2disk_(obj):
        CACHE_PATH = './cache'
        fops.make_dir(CACHE_PATH)

        cache_fname = obj.__module__ + obj.__name__
        cachepath = fops.construct_path(cache_fname, CACHE_PATH)

        def dump_cache():
            cach_new_str = cPickle.dumps(cache, 2)
            # If the cache changed in this run, save it - will have to
            # compare dicts properly
            #if hash(cache_str) != hash(cach_new_str):
            print('writing memoization cache to disk...')
            fops.write_data(cachepath, cach_new_str)

        try:
            print('loading memoization cache from disk...')
            cache_str = fops.get_data(cachepath)
            cache = cPickle.loads(cache_str)
            #cache = {}
        except:
            #print('Could not open cache file!')
            # Every object should have its own cache?
            # Or one for all functions accrosss all objects of the class?
            #cache = obj.cache = {}
            cache = {}
            cache_str = ''

        atexit.register(dump_cache)

        #@functools.wraps(obj)
        def memoizer(*args, **kwargs):
            key = hash_fun(args, kwargs)
            if key not in cache:
                cache[key] = obj(*args, **kwargs)
            return cache[key]
        return memoizer
    return memoize2disk_


# Understand this better....
# # https://gist.github.com/codysoyland/267733/
# class Memoize2Disk(object):
#     def __init__(self, func):
#         self.func = func
#         self.memoized = {}
#         self.method_cache = {}
#     def __call__(self, *args):
#         return self.cache_get(self.memoized, args,
#             lambda: self.func(*args))
#     def __get__(self, obj, objtype):
#         return self.cache_get(self.method_cache, obj,
#             lambda: self.__class__(functools.partial(self.func, obj)))
#     def cache_get(self, cache, key, func):
#         try:
#             return cache[key]
#         except KeyError:
#             cache[key] = func()
#             return cache[key]

def memoize2mem(obj):
    """memoize2
    Notes
    ------
    Memoize functions. Do not use with object methods or class methods
    as it will not save associated variables.
    Good function to memoize without using disk
    """
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = my_fast_str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


def assert_no_Nones(self):
    for k, d in self.__dict__:
        if d is None:
            raise err.Fatal('undefined attribute: {}'.format(k))


# flattens lists with 1 level nesting.
# Replaces empty nested lists with None to preserve list lengths
def flatten(l):
    l = [i if i else [None] for i in l]
    return [x for y in l for x in y]


# cumulitive sum of all elements
# NOTE: makes the first element 0!
def cumsum(lst):
    ret_val = [0]
    s = 0
    for i in lst:
        s += i
        ret_val.append(s)
    return ret_val


# lst: list
# num_e_list: a list of numbers of elements dictating the grouping
# ret_val satisfies = [length(sub_list) for sub_list in ret_val] == num_e_list
#
# e.g.: list = [1,2,3,4,5], num_e_list = [3,2]
# ret_val = [[1,2,3], [4,5]]
# groups the list elements to vreate a level 1 nested list using the idx_list
def group_list(lst, num_e_list):
    idx_list = cumsum(num_e_list)
    # get the slice list
    slice_list = [slice(s, e) for s, e in pairwise(idx_list)]
    return [lst[s] for s in slice_list]


def bounded_iter(iter_obj, MAX):
    for i in range(MAX):
        yield iter_obj.next()


def while_max_iter(MAX):
    ctr = [0]

    def f(cond):
        ctr[0] += 1
        #if ctr > MAX: return False else: return cond
        return not(ctr[0] > MAX) and cond
    return f


# ################### Unfinished
class PrintSteady():
    def __init__(self):
        self.t = Terminal()
        print('')
        print(self.t.move_up + self.t.move_up)

    def p(self, msg):
        with self.t.location():
            print(str(msg))


def demo_print_steady():
    print('num:', end='')
    for i in range(10):
            PrintSteady(str(i))
            time.sleep(0.1)
    print('')
# ################### /Unfinished


def pause(msg=''):
    prompt = 'press enter to continue...'
    if msg != '':
        prompt = '{}: {}'.format(msg, prompt)
    raw_input(prompt)


def inf_list(x):
    while True:
        yield x


def eprint(*args, **kwargs):
    """prints messages on stderror"""
    return print(*args, file=sys.stderr, **kwargs)


def print(*args, **kwargs):
    """custom print() function."""
    # A hack. Look inside and if args is empty, do not do anything,
    # just call the default print function.
    if args:
        callers_frame_idx = 1
        (frame, filename, lineno,
         function_name, lines, index) = inspect.getouterframes(
                                             inspect.currentframe())[callers_frame_idx]
        #frameinfo = inspect.getframeinfo(inspect.currentframe())
        basename = fops.get_file_name_from_path(filename)
        f = kwargs.get('file', sys.stdout)
        __builtin__.print('{}:{}::'.format(basename, lineno), end='', file=f)
    return __builtin__.print(*args, **kwargs)


class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, max_time):
        self.max_time = max_time

    def handle_timeout(self, signum, frame):
        raise TimeoutError('Timed Out: t > {}'.format(self.max_time))

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.max_time)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# TODO: Eventually, move it somehwere more sensible
def poly_sat(poly, x, tol=0):
    """poly_sat

    Parameters
    ----------
    poly : polytope
    x : vector

    Returns True if x is a member of polytope p
    """
#     print poly.C
#     print poly.d
#     print x
    import numpy as np
    return np.all(np.dot(poly.C, x) <= poly.d)


# TODO: relocate to a better file
@memoize2mem
def poly_v2h(x):
    hull = ConvexHull(x)
    eqns = hull.equations
    C, d = eqns[:, 0:-1], eqns[:, -1]
    return C, d


def ceil(x, ceil_fn):
    """A ceil function which tries to detect floating point issues

    For Ex. np.ceil(.2*3/.2) = 4
        But should be 3!
        The detection is done by using a tolerance to estimate that a
        floating point error has likely occured
    """
    tol = 0.9999999

    # if the difference between x and ceil(x) is more than ~ 1, error
    # has occured
    ceil_x = ceil_fn(x)
    error = ceil_x - x > tol
    return (ceil_x - 1 if error else ceil_x)


def invert_mapping(d):
    if __debug__:
        items = d.items()
        # Make sure duplicate mappings do not exist
        assert(len(list(items)) == len(set(items)))

    return d.__class__(map(reversed, d.items()))


def dict_unique_add(d, k, data):
    """If the key is already present, raise an exception

    Parameters
    ----------
    d : dict
    k : key
    data : data
    """

    if __debug__:
        # If overwrite with different value occurs
        if k in d:
            try:
                if hash(d[k]) != hash(data):
                    raise RuntimeError
            except TypeError:
                if d[k] != data:
                    raise RuntimeError

    d[k] = data

    return None
