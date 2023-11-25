import numpy as np
import pandas as pd

def myfft(x):
    N = len(x)
    if N <= 1:
        return x
    evens = myfft(x[0::2])
    odds = myfft(x[1::2])
    t = np.exp(-2j * np.pi * np.arange(N) / N)
    f = np.concatenate([evens + t[:N//2] * odds,
                        evens + t[N//2:] * odds])
    return f

def myifft(x):
    N = len(x)
    if N <= 1:
        return x
    evens = myifft(x[0::2])
    odds = myifft(x[1::2])
    t = np.exp(2j * np.pi * np.arange(N) / N)
    f = np.concatenate([evens + t[:N//2] * odds,
                        evens + t[N//2:] * odds])
    return f / 2

def main():
    data = np.random.normal(size=1024)

    ffts1 = np.fft.fft(data)
    iffts1 = np.fft.ifft(ffts1)

    ffts2 = np.fft.fft(data)
    iffts2 = np.fft.ifft(ffts2)

    print(np.allclose(ffts1, ffts2))

main()
