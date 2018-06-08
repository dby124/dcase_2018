#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 beatrice <beatrice@tju.edu.cn>
#
# Distributed under terms of the DSPLAB in Tianjin University license.

"""
Hilbert transform
"""

import numpy as np
from pyhht.emd import EMD
from pyhht.utils import inst_freq
from numpy.matlib import repmat
from scipy.signal import hilbert


def get_hht(x, fs):
    """
    Hilbert-Huang transform of a frame in signal
    :param x:   the frame in signal
    :param fs:
    :return:
    A: numpy.ndarray, shape:(n_imf, len(x)- 1), n_imf means the number of imfs.
    f: numpy.ndarray,shape:(n_imf, len(x)- 1), n_imf means the number of imfs.
    bjp: numpy.ndarray, shape:len(x)- 1.
    """
    t = np.linspace(0, len(x) - 1, len(x) - 1)
    decomposer = EMD(x)
    imfs = decomposer.decompose()

    A, f, tt = hhspectrum(imfs)

    E, tt1, ff = toimage(A, f, tt, len(tt))
    bjp = getbjp(E, fs)  # calculate Marginal spectrum
    return A, f, bjp


def hhspectrum(x, t=None, start_l=0):
    """
    [A,f,tt] = HHSPECTRUM(x,t,l) computes the Hilbert-Huang spectrum
    :param x: matrix with one signal per row
    :param t: time instants
    :param start_l: estimation parameter for instfreq (integer >=0 (0:default))
    :type x: numpy.ndarray
    :type t: numpy.ndarray
    :type start_l: int
    :return:
        - A   : instantaneous amplitudes
        - f   : instantaneous frequencies
        - tt  : truncated time instants
    :type A: numpy.ndarray
    :type f: numpy.ndarray
    :type tt: int
    :calls:
      - hilbert  : computes the analytic signal
      - inst_freq : computes the instantaneous frequency
    :Examples:
    s = randn(1,512);
    imf = emd(s);
    [A,f,tt] = hhspectrum(imf(1:end-1,:));

    s = randn(10,512);
    [A,f,tt] = hhspectrum(s,1:512,2,1);

    rem: need the Time-Frequency Toolbox (http://tftb.nongnu.org)

    See also
     emd, toimage, disp_hhs

    G. Rilling, last modification 3.2007
    gabriel.rilling@ens-lyon.fr

    .. plot:: docstring_plots/utils/inst_freq.py
    noted: this code transplanted from matlab.
    """
    if t is None:
        t = range(0, x.shape[1])

    x = np.array(x)
    if min(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.T

    lt = len(t)
    t = np.array(t)
    tt = t[(start_l + 1):(lt - start_l)]

    an = hilbert(x[0, :].T)
    f, time = inst_freq(an, tt, start_l)
    k = 0

    for item in x:
        if k > 0:
            an_temp = hilbert(item.T)
            f_temp, time_temp = inst_freq(an_temp, tt, start_l)
            an = np.vstack((an, an_temp))
            f = np.vstack((f, f_temp))
        k = k + 1

    A = abs(an[:, (start_l + 1): len(an[0]) - start_l])

    return A, f, tt


def accumarray(subs, val, *args):
    """
    Construct array with accumulation
    :param subs:
    :param val:
    :param args:
    :return:
    """

    subs = np.array(subs)
    val = np.array(val)
    # define marx, if sz is null, the value will be given automatically.
    if len(args) == 0:
        marx = np.zeros([max(subs.T[0]) + 1, max(subs.T[1]) + 1], float)

    else:
        marx = np.zeros(np.array(args[0]), float)

    for index in range(len(subs[0])):
        indx = subs[0][index]
        indy = subs[1][index]
        marx[indx][indy] = marx[indx][indy] + val[index]

    return marx


def toimage(A, f, *args):
    """

    :param A: instantaneous amplitudes (amplitudes of modes (1 mode per row of A))
    :param f: instantaneous frequencies
    :param args: The dimensions of the intput matrix, include truncated time instants and its length
    :type A: numpy.ndarray
    :type f: numpy.ndarray
    :type args: numpy.ndarray
    :return:
        - E: 2D image of the spectrum
        - tt1: time instants in the image
        - ff: centers of the frequency bins
    :type E: numpy.ndarray
    :type tt1: numpy.ndarray
    :type ff: numpy.ndarray
    :calls:
      - accumarray  : Construct array with accumulation
    """
    global sply
    global splx
    DEFSPL = 400

    if (len(args) < 0) or (len(args) > 3):
        raise NameError('Input parameters do not meet requirements')
    elif len(args) == 0:
        tt = np.linspace(0, A.shape[1], A.shape[1])
        sply = DEFSPL
        splx = len(tt)
    elif len(args) == 1:
        if np.isscalar(args[0]):
            tt = np.linspace(0, A.shape[1], A.shape[1])
            splx = len(tt)
            sply = args[0]
        else:
            tt = args[0]
            splx = len(tt)
            sply = DEFSPL
    elif len(args) == 2:
        if np.isscalar(args[0]):
            tt = np.linspace(0, A.shape[1], A.shape[1])
            sply = args[0]
            splx = args[1]
        else:
            tt = args[0]
            sply = args[1]
            splx = len(tt)
    elif len(args) == 3:
            tt = args[0]
            splx = args[1]
            sply = args[2]

    if len(A.shape) < 2:
        A = A.T
        f = f.T

    # Here Omitted for verifying the validity of the parameters
    f = np.where(f < 0.5, f, 0.5)
    f = np.where(f > 0, f, 0)

    indf = np.round(2*f*(sply-1))
    indt = repmat(np.round(np.linspace(0, len(tt)-1, splx)), A.shape[0], 1)

    indf = indf.astype(int)
    indt = indt.astype(int)

    E = accumarray([indf.T.flatten(), indt.T.flatten()], A.T.flatten(), [sply, splx])

    indt = indt[1, :]
    tt2 = tt[indt]
    ff = np.round(np.linspace(0, sply-1, sply))*0.5/sply + (1/(4*sply))
    return E, tt2, ff


def getbjp(E, fs):
    """
    calculate Marginal spectrum
    :param E:
    :param fs:
    :return:
    """
    E = np.flipud(E)

    row = len(E.T[0])
    bjp = np.zeros(row)
    for k in range(row):
        temp = sum(E[k]) / fs
        bjp[k] = temp

    return bjp

