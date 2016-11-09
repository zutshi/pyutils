#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import itertools as it

import numpy as np

logger = logging.getLogger(__name__)


class ConstraintsError(Exception):
    pass


class Constraints(object):

    def __init__(self):
        pass


def top2ic(n):
    """Get interval constraints equivalent to the Universal set

    Parameters
    ----------
    n : dimension

    Returns
    -------
    IntervalCons([[-inf..-inf], [inf..inf]])
    """
    return IntervalCons([-np.inf]*n, [np.inf]*n)


def zero2ic(n):
    """Get IntervalCons of n-dim, s.t., xi = 0

    Parameters
    ----------
    n : dimension

    Returns
    -------
    IntervalCons([[0..0], [0..0]])
    """
    return IntervalCons([0.0]*n, [0.0]*n)


# Add multiple init constructors using kwargs
class IntervalCons(Constraints):
    @staticmethod
    def dot(iv1, iv2):
        iv1_arr, iv2_arr = iv1.to_numpy_array(), iv2.to_numpy_array()
        prod = np.asarray([np.prod(list(it.product(ri, rj)), axis=1) for ri, rj in zip(iv1_arr, iv2_arr)])
        return IntervalCons(np.min(prod, axis=1), np.max(prod, axis=1))

    @staticmethod
    def concatenate(ic1, ic2):
        return IntervalCons(
                np.hstack((ic1.l, ic2.l)),
                np.hstack((ic1.h, ic2.h))
                )

    @staticmethod
    def from_array_like(x):
        """Builds a interval constraint from a structure resembling
        [xa = [x0l, x1l, x2l, ...], xb = [x0h, x1h, x2h, ...]]

        Parameters
        ----------
        x : array like iterable

        Returns
        -------
        IntervalCons(xl, xh)

        Notes
        ------
        Automatically finds xil = min(xai, xbi), xih = max(xai, xbi)
        in order to build a proper interval constraint reprsentation.
        """
        xa, xb = x
        temp = [(min(xai, xbi), max(xai, xbi)) for xai, xbi in zip(xa, xb)]
        # Take the transpose
        xl, xh = zip(*temp)
        return IntervalCons(xl, xh)

    def __init__(self, l, h, sanity_check=True):
        #if type(h) != np.ndarray or type(l) != np.ndarray:
        try:
            self.l = np.asarray(l)
            self.h = np.asarray(h)
        except:
            raise ConstraintsError(
                    'interval constraints should be convertible to numpy arrays'
                    )
        self.dim = self.l.size
        if sanity_check:
            self.sanity_check()

    # sanity check
    def sanity_check(self):
        if not (self.l.ndim == 1 and self.h.ndim == 1):
            raise ConstraintsError('dimension is not 1!')

        if self.l.size != self.h.size:
            raise ConstraintsError('dimension mismatch!')

        if not all(self.l <= self.h):
            raise ConstraintsError('malformed interval: l:{}, h:{}'.format(self.l, self.h))

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
        s = [(round(self.l[i], 2), round(self.h[i], 2)) for i in range(self.dim)]
        return str(s)

    def any_sat(self, x_array):
        return np.logical_or.reduce(self.sat(x_array), 0)

    # vecotirzed version of __contains__()
    def sat(self, x_array):
        """sat

        Parameters
        ----------
        x_array : array of state vectors
        """
        res_l = x_array >= self.l
        res_h = x_array <= self.h

        # Columnwise reduction
        land = np.logical_and(res_l, res_h)
        if land.ndim > 1:
            return np.logical_and.reduce(land, 1)
        else:
            return land

    def contains(self, ic2):
        """ Does ic contain ic2?

        Parameters
        ----------
        ic2 : interval cons

        Returns
        -------
        Bool: ic \supseteq \ic2 ?
        """
        return np.all(self.h >= ic2.h) and np.all(self.l <= ic2.l)

    def __and__(self, ic):
        """ overloaded & """

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

    def __neg__(self):
        return IntervalCons(-self.h, -self.l)

    def __add__(self, c):
        if isinstance(c, IntervalCons):
            l, h = self.l + c.l, self.h + c.h
        else:
            l, h = self.l + c, self.h + c
        return IntervalCons(l, h)

    def __radd__(self, c):
        return self + c

    def __sub__(self, c):
        return self - c

    def __rsub__(self, c):
        return -self + c

    #dot product with a numpy vector c
    def __mul__(self, c):
        iv = self
        c = np.asarray(c)

        temp = iv.to_numpy_array().T * c

        # only used to check assertion
        def post_mult():
            if c.size == 1:
                if c < 0:
                    prod = IntervalCons(temp[1], temp[0])
                else:
                    prod = IntervalCons(temp[0], temp[1])
            else:
                for idx, i in enumerate(c):
                    # if i is -ve, swap
                    if i < 0:
                        temp[0][idx], temp[1][idx] = temp[1][idx], temp[0][idx]
                prod = IntervalCons(temp[0], temp[1])
            return prod

        prod = IntervalCons.from_array_like(temp)
        assert(prod == post_mult())
        return prod

    def __rmul__(self, c):
        return self * c

    @property
    def zero_measure(self):
        return np.any(self.l - self.h == 0)

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
        return '[{},{}]'.format(str(self.l), str(self.h))


#        s = '[' + str(self.l) + ',' + str(self.h) + ']'
#        return s
