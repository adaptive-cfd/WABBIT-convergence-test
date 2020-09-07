#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:54:50 2020

@author: https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
"""
from math import floor, log10


# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num == 0:
        return "$0$"
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    if abs(exponent)>2:
        return r"${0:.{2}f}\times 10^{{{1:d}}}$".format(coeff, exponent, precision)
    else:
        return "$"+str(num)+"$"