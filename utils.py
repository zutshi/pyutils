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
from blessed import Terminal
import signal

import err
import fileops as fops

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
        raise err.Fatal('call() returned non-zero return value')


# subprocess.check_output

def strict_call_get_op(*args, **kwargs):
    try:
        op = subprocess.check_output(stderr=subprocess.STDOUT, *args, **kwargs)
    except subprocess.CalledProcessError, e:
        term = Terminal()

        # print term.red(e.output)
#        error_msg = str(e.output).decode('utf-8')
#        print term.red(error_msg)

        error_msg = e.output.decode('utf-8')
        print(term.red(error_msg))

        # print term.red(str(e.output))

        sys.stdout.flush()
        raise

    # TODO: what is op?
    # print op

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
    return '=' * 20 + '\n' + str(s) + '\n' + '=' * 20


def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


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


#################### Unfinished
class PrintSteady():
    def __init__(self):
        self.t = Terminal()
        print('')
        print(self.t.move_up + self.t.move_up)

    def p(self, msg):
        with self.t.location():
            print(str(msg))


def demo_print_steady():
    import time
    print('num:', end='')
    for i in range(10):
            PrintSteady(str(i))
            time.sleep(0.1)
    print('')
#################### /Unfinished


def pause(msg=''):
    prompt = 'press enter to continue...'
    if msg != '':
        prompt = '{}: {}'.format(msg, prompt)
    raw_input(prompt)

def inf_list(x):
    while True:
        yield x

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
