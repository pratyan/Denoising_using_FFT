import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16,12]
plt.rcParams.update({'font.size': 18})

# Creating a simple signal with two frequencies
dt = 0.001
t = np.arange(0,1,dt)

#signal
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) 

#clean signal
f_clean = f

#signal with noise
f_noise = f + 2.5*np.random.randn(len(t))


# computing the FFT

n = len(t)
fhat = np.fft.fft(f_noise,n)               # compute the fft
PSD = fhat * np.conj(fhat) / n       # Power spectrum (power per freq)

#creating x-axis of freq
freq = (1/(dt*n)) * np.arange(n)

#only plot the first half
L = np.arange(1,np.floor(n/2),dtype='int')


# using the PSD to filter out noise
indices = PSD > 100      # find out freqs with large power
PSDclean = PSD * indices # zero out all others
fhat = indices * fhat    # zero out small Fourier coeffs in Y

# now, computing the filtered signal by inverse fft of the removed coeff
ffilt = np.fft.ifft(fhat)
