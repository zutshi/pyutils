#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import inspect

from blessed import Terminal

term = Terminal()


class Fatal(Exception):

    pass


# def error(msg):
#     print term.red('ERROR: ' + msg)
#     exit(-1)

WARN_STR = '{} :{}::WARNING: {}'


def warn(msg):
    callers_frame_idx = 1
    (frame, filename, lineno,
     function_name, lines, index) = inspect.getouterframes(
                                         inspect.currentframe())[callers_frame_idx]
    msg_ = WARN_STR.format(filename, lineno, msg)
    print(term.red_on_white(msg_))


def warn_severe(msg):
    callers_frame_idx = 1
    (frame, filename, lineno,
     function_name, lines, index) = inspect.getouterframes(
                                         inspect.currentframe())[callers_frame_idx]
    msg_ = WARN_STR.format(filename, lineno, msg)
    print(term.red_on_white(msg_))
    # forces the user to take heed!
    raw_input('please acknowledge by pressing enter')


# def int_error(msg):
#     print 'INTERNAL ERROR: ' + msg
#     exit(-1)
