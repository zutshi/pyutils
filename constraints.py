#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ConstraintsError(Exception):
    pass


class Constraints(object):

    def __init__(self):
        pass


def top2ic(n):
    return IntervalCons([-np.inf]*n, [np.inf]*n)


class IntervalCons(Constraints):
    @staticmethod
    def concatenate(ic1, ic2):
        return IntervalCons(
                np.hstack((ic1.l, ic2.l)),
                np.hstack((ic1.h, ic2.h))
                )

    def __init__(self, l, h):
        #if type(h) != np.ndarray or type(l) != np.ndarray:
        try:
            self.l = np.asarray(l)
            self.h = np.asarray(h)
        except:
            raise ConstraintsError(
                    'interval constraints should be convertible to numpy arrays'
                    )
        self.dim = self.l.size
        self.sanity_check()

    # sanity check
    def sanity_check(self):
        if not (self.l.ndim == 1 and self.h.ndim == 1):
            raise ConstraintsError('dimension is not 1!')

        if self.l.size != self.h.size:
            raise ConstraintsError('dimension mismatch!')

        if not all(self.l <= self.h):
            raise ConstraintsError('malformed interval!')

    def scaleNround(self, CONVERSION_FACTOR):
        raise ConstraintsError('#$%^$&#%#&%$^$%^$^#!@$')

        # do not do inplace conversion, instead return a copy!
        # self.h = (self.h * CONVERSION_FACTOR).astype(int)
        # self.l = (self.l * CONVERSION_FACTOR).astype(int)

        h = np.round(self.h * CONVERSION_FACTOR).astype(int)
        l = np.round(self.l * CONVERSION_FACTOR).astype(int)
        return IntervalCons(l, h)

    # Returns the more traditional format for expressing constraints
    # [[x1_l, x1_h],
    #  [x2_l, x2_h],
    #  [x3_l, x3_h]]
    def to_numpy_array(self):
        return np.concatenate(([self.l], [self.h]), axis=0).T

    # Get the constraints as C program statements!
    # [(1,10), (2,12)]
    # will become
    # [
    #   'x[0] >= 1',
    #   'x[0] <= 2'.
    #   'x[1] >= 10'
    #   'x[1] <= 12
    # ]

    def to_c_str_list(self, var_name):

        # template string

        ts_ge = '{0}[{1}] >= {2}'
        ts_le = '{0}[{1}] <= {2}'
        s = []
        for (idx, l, h) in zip(range(len(self.l)), self.l, self.h):
            s.append(ts_ge.format(var_name, idx, l))
            s.append(ts_le.format(var_name, idx, h))
        return s

    def __str__(self):
        s = self.to_c_str_list('x')
        return ' and '.join(s)

    def any_sat(self, x_array):
        return np.logical_or.reduce(self.sat(x_array), 0)

    def sat(self, x_array):
        res_l = x_array >= self.l
        res_h = x_array <= self.h

        # Columnwise reduction

        return np.logical_and.reduce(np.logical_and(res_l, res_h), 1)

    def __and__(self, ic):

        # check dimensions

        if self.dim != ic.dim:
            raise ConstraintsError(
                    'intersection of interval constraints: must be of same dimensions!'
                    )

#       use max/min instead of case check!

        l = np.maximum(self.l, ic.l)
        h = np.minimum(self.h, ic.h)
        try:
            return IntervalCons(l, h)
        except ConstraintsError:
            return None

    # must return True or False
    # Use contains() for vectorized function

    def __contains__(self, x):

        if x.ndim != 1:
            raise ConstraintsError('dim(x) must be 1, instead dim(x) = {}'.format(x.ndim))

        # print x.shape, x

        res_l = x >= self.l
        res_h = x <= self.h

        # print np.logical_and.reduce(np.logical_and(res_l, res_h), 0)

        ret_val = np.logical_and.reduce(np.logical_and(res_l, res_h), 0)
        return ret_val

    # vecotirzed version of __contains__()

    def contains(self, x_array):
        """contains

        Parameters
        ----------
        x_array : array of state vectors

        Returns
        -------
        TODO?
        """

        # print x.shape, x

        res_l = x_array >= self.l
        res_h = x_array <= self.h

        # The axis used is different from __contains__()

        return np.logical_and.reduce(np.logical_and(res_l, res_h), 1)

    def poly(self):
        '''converts ival cons to polyhedral constrains.
           Returns C, d, such that Cx <= d.
           C is a nxn array and d is a 1xn array'''

        # C = [[I], [-I]]
        C = np.vstack([np.eye(self.dim), -np.eye(self.dim)])
        d = np.hstack([self.h, -self.l])
        return C, d

    def smt2(self, x, smt_And):
        """smt2

        Parameters
        ----------
        x : list of smt vars
        smt_And : smt And function

        Returns
        -------
        Smt Expression
        """
        ic = self
        cons_list = (map((lambda x, c: x <= c), x, ic.h) +
                     map((lambda x, c: x >= c), x, ic.l)
                     )
        return smt_And(cons_list)

    def rect(self):
        """Converts to rectangular representation.
           Returns: (corner, length)
           Useful for plotting with matplotlib
        """
        return self.l, (self.h - self.l)

    def sample_UR(self, N):
        random_arr = np.random.rand(N, self.dim)
        x_array = self.l + random_arr * (self.h - self.l)
        return x_array

    #TODO: Might be slow...
    def __hash__(self):
        return hash(tuple(np.concatenate((self.l, self.h))))

    def __eq__(self, ic):
        assert(isinstance(ic, IntervalCons))
        return all(ic.l == self.l) and all(ic.h == self.h)

    def __repr__(self):
        return '{},{}'.format(str(self.l), str(self.h))


#        s = '[' + str(self.l) + ',' + str(self.h) + ']'
#        return s
