#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 此处为新加特征提取类，继承于FeatureExtractor
#
# Copyright © 2018 beatrice <beatrice@tju.edu.cn>
#
# Distributed under terms of the DSPLAB in Tianjin University license.


# from __future__ import print_function, absolute_import

import numpy as np
from librosa import util
import scipy
from dcase_util.features import FeatureExtractor
from dcase_util.ui import FancyStringifier
from hht_extension import get_hht


class HHTFeatureExtractor(FeatureExtractor):
    """HHT related feature extractor base class"""
    def __init__(self, fs=48000, win_length_samples=None, hop_length_samples=None, win_length_seconds=0.04,
                 hop_length_seconds=0.02, n_imf=4, spectrogram_type='all', n_hht=1920, window_type='hamming_asymmetric',
                 **kwargs):
        """Constructor

        Parameters
        ----------
        fs : int
            Sampling rate of the incoming signal.

        win_length_samples : int
            Window length in samples.

        hop_length_samples : int
            Hop length in samples.

        win_length_seconds : float
            Window length in seconds.

        hop_length_seconds : float
            Hop length in seconds.

        spectrogram_type : str
            Spectrogram type, magnitude or power spectrogram.
            Default value 'magnitude'

        n_hht : int
            Length of the HHT window.
            Default value 2048

        window_type : str
            Window function type.
            Default value 'hamming_asymmetric'

        """

        super(HHTFeatureExtractor, self).__init__(**kwargs)

        # Run FeatureExtractor init
        FeatureExtractor.__init__(self, **kwargs)

        self.fs = fs
        self.win_length_samples = win_length_samples
        self.hop_length_samples = hop_length_samples

        self.win_length_seconds = win_length_seconds
        self.hop_length_seconds = hop_length_seconds

        if not self.win_length_samples and self.win_length_seconds and self.fs:
            self.win_length_samples = int(self.fs * self.win_length_seconds)

        if not self.hop_length_samples and self.hop_length_seconds and self.fs:
            self.hop_length_samples = int(self.fs * self.hop_length_seconds)

        if self.win_length_samples is None:
            message = '{name}: No win_length_samples set'.format(
                name=self.__class__.__name__
            )
            self.logger.exception(message)
            raise ValueError(message)

        if self.hop_length_samples is None:
            message = '{name}: No hop_length_samples set'.format(
                name=self.__class__.__name__
            )
            self.logger.exception(message)
            raise ValueError(message)
        self.n_imf = n_imf
        self.spectrogram_type = spectrogram_type
        self.n_hht = n_hht
        self.window_type = window_type

        self.window = self.get_window_function(
            n=self.win_length_samples,
            window_type=self.window_type
        )

    def __str__(self):
        ui = FancyStringifier()
        output = super(HHTFeatureExtractor, self).__str__()

        output += ui.data(indent=4, field='n_imf', value=self.n_imf) + '\n'
        output += ui.line(field='Spectrogram') + '\n'
        output += ui.data(indent=4, field='spectrogram_type', value=self.spectrogram_type) + '\n'
        output += ui.data(indent=4, field='n_hht', value=self.n_hht) + '\n'
        output += ui.data(indent=4, field='window_type', value=self.window_type) + '\n'

        return output

    def get_window_function(self, n, window_type='hamming_asymmetric'):
        """Window function

        Parameters
        ----------
        n : int
            window length

        window_type : str
            window type
            Default value 'hamming_asymmetric'

        Raises
        ------
        ValueError:
            Unknown window type

        Returns
        -------
        numpy.array
            window function

        """

        # Windowing function
        if window_type == 'hamming_asymmetric':
            return scipy.signal.hamming(n, sym=False)

        elif window_type == 'hamming_symmetric' or window_type == 'hamming':
            return scipy.signal.hamming(n, sym=True)

        elif window_type == 'hann_asymmetric':
            return scipy.signal.hann(n, sym=False)

        elif window_type == 'hann_symmetric' or window_type == 'hann':
            return scipy.signal.hann(n, sym=True)

        else:
            message = '{name}: Unknown window type [{window_type}]'.format(
                name=self.__class__.__name__,
                window_type=window_type
            )

            self.logger.exception(message)
            raise ValueError(message)

    def hht_based_feature(self, A, f, bjp):
        """
        Extract features based Hilbert-Huang transform (HHT)

        A: numpy.ndarray, shape:(n_imf, len(x)- 1), n_imf means the number of imfs.
        f: numpy.ndarray,shape:(n_imf, len(x)- 1), n_imf means the number of imfs.
        bjp: numpy.ndarray, shape:len(x)- 1.

        Parameters
        ----------
        A : np.ndarray [shape=(n_imf, n_hht-1)], instantaneous amplitude frequency,
            n_imf means the number of imfs.

        f : np.ndarray [shape=(n_imf, n_hht-1)], instantaneous frequency,
            n_imf means the number of imfs.

        bjp  : np.ndarray [shape=(n_hht-1,), dtype=dtype]

        Returns
        -------
        hht_features : np.ndarray [30,), dtype=dtype]

        """
        hht_features = np.empty(30, dtype=np.complex64, order='F')
        hht_features[8] = np.var(bjp)

        # 边际谱的带宽和衰减个数（即在matlab中的num-band）
        for k in range(len(bjp)):
            if bjp[k] > 1e-6:
                hht_features[10] = hht_features[10] + 1
            if bjp[k] > 5e-7:
                hht_features[11] = hht_features[11] + 1

        hht_features[11] = hht_features[11] - hht_features[10]

        # 序号为0-9的特征，代表边际谱的前五个峰值及其对应的位置
        for item in range(5):
            hht_features[item] = np.max(bjp)
            hht_features[5 + item] = np.argmax(bjp)
            bjp[np.argmax(bjp)] = 0

        # 边际谱方差
        hht_features[12] = np.var(bjp)

        # if A.shape[0] < 5:
        #     print("out of range:", A.shape[0])

        # 前6个分量瞬时频率的均值，方差和有效值
        for k in range(5):
            hht_features[13 + k] = np.mean(A[k, :])
            hht_features[18 + k] = np.var(A[k, :])
            hht_features[23 + k] = np.sqrt(sum(np.power((A[k, :] - np.mean(A[k, :])), 2)) / A.shape[0])

        # 所有分量瞬时频率的方差和
        for k in range(f.shape[0]):
            if k == 0:
                hht_features[28] = np.var(f[k, :])

            hht_features[29] = hht_features[29] + np.var(f[k, :])   # 所有分量瞬时频率的方差和

        hht_features[28] = hht_features[28] / hht_features[29]      # 第一个分量的瞬时频率方差贡献率

        return hht_features

    def hht(self, y, hop_length=None, win_length=None,
            center=True, dtype=np.complex64, pad_mode='reflect'):
        """Hilbert-Huang transform (HHT)

        Parameters
        ----------
        y : np.ndarray [shape=(n,)], real-valued
            the input signal (audio time series)

        hop_length : int > 0 [scalar]
            number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.

            If unspecified, defaults to ``win_length = n_fft``.

        center      : boolean
            - If `True`, the signal `y` is padded so that frame
              `D[:, t]` is centered at `y[t * hop_length]`.
            - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

        dtype       : numeric type
            Complex numeric type for `D`.  Default is 64-bit complex.

        pad_mode : string
            If `center=True`, the padding mode to use at the edges of the signal.
            By default, HHT uses reflection padding.

        Returns
        -------
        hht_matrix : np.ndarray [shape=(30, t), dtype=dtype]
        bjp_matrix : np.ndarray [shape=(n_hht-1, t), dtype=dtype]

        """

        # By default, use the entire frame
        if win_length is None:
            win_length = self.n_hht

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length / 2)

        hht_window = self.window

        # Pad the window out to n_hht size
        hht_window = util.pad_center(hht_window, self.n_hht)

        # Reshape so that the window can be broadcast
        hht_window = hht_window.reshape((-1, 1))

        # Check audio is valid
        util.valid_audio(y)

        # Pad the time series so that frames are centered
        if center:
            y = np.pad(y, self.n_hht - 1, mode=pad_mode)

        # Window the time series.
        y_frames = util.frame(y, frame_length=self.n_hht, hop_length=hop_length).T

        # Pre-allocate the HHT matrix
        hht_matrix = np.empty((30, y_frames.shape[0]),
                              dtype=dtype,
                              order='F')

        bjp_matrix = np.empty((self.n_hht - 1, y_frames.shape[0]),
                              dtype=dtype,
                              order='F')

        for bl_s in range(hht_matrix.shape[1]):
            frame_signal = hht_window[:, 0] * y_frames[bl_s, :]
            A, f, bjp = get_hht(frame_signal, self.fs)
            hht_matrix[:, bl_s] = self.hht_based_feature(A, f*self.fs, bjp)
            bjp_matrix[:, bl_s] = bjp

        return hht_matrix, bjp_matrix

    def get_hhtspectrogram(self, y, spectrogram_type=None):
        """Spectrogram

        Parameters
        ----------
        y : numpy.ndarray
            Audio data

        spectrogram_type : str
            Type of spectrogram "all" or "marginal_spectrum"
            Default value None

        Returns
        -------
        numpy.ndarray [shape=(n_fft-1, t)] when 'spectrogram_type'='marginal_spectrum'
         or [shape=(30, t)] when 'spectrogram_type'='all'
            spectrum

        """

        if spectrogram_type is None:
            spectrogram_type = self.spectrogram_type

        from dcase_util.containers import AudioContainer

        if isinstance(y, AudioContainer):
            if y.channels == 1:
                y = y.data

            else:
                message = '{name}: Input has more than one audio channel.'.format(
                    name=self.__class__.__name__
                )

                self.logger.exception(message)
                raise ValueError(message)

        hht_matrix, bjp_matrix = self.hht(
            y + self.eps,
            hop_length=self.hop_length_samples,
            win_length=self.win_length_samples
        )

        if spectrogram_type == 'all':
            return hht_matrix
        elif spectrogram_type == 'marginal_spectrum':
            return bjp_matrix
        else:
            message = '{name}: Unknown spectrum type [{spectrogram_type}]'.format(
                name=self.__class__.__name__,
                spectrogram_type=spectrogram_type
            )

            self.logger.exception(message)
            raise ValueError(message)

    def extract(self, y):
        """Extract features for the audio signal.

        Parameters
        ----------
        y : AudioContainer or numpy.ndarray [shape=(n,)]
            Audio signal

        Returns
        -------
        numpy.ndarray [shape=(n_fft-1, t)] when 'spectrogram_type'='marginal_spectrum'
         or [shape=(30, t)] when 'spectrogram_type'='all'
            spectrum
        """

        return self.get_hhtspectrogram(y=y, spectrogram_type=self.spectrogram_type)

