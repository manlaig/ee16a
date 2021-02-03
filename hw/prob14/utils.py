import numpy as np
from numpy.fft import fft, ifft
import scipy.io.wavfile
from IPython.display import Audio, display
import warnings

warnings.filterwarnings("ignore")


RATE = 44100

def load_recording(filename):
    return scipy.io.wavfile.read(filename)[1].astype("float64")

def save_recording(filename, recording):
    scipy.io.wavfile.write(filename, RATE, recording)
    
def play(data):
    display(Audio(data=data, rate=RATE, autoplay=False))

def cross_correlate(x, y):
    return ifft(fft(x) * fft(y).conj()).real
    
def omp(A, b, k):
    assert isinstance(A, np.array)
    assert isinstance(b, np.array)
    assert isinstance(k, int)
    m, n = A.shape
    assert len(b) == n, "b is the wrong shape"
    
    out = np.zeros(m)
    residual = b
    
    vectors_used = []
    
    for _ in range(k):
        inner_products = A.T.dot(residual)
        best_candidate = A[:, np.argmax(inner_products)]
        vectors_used.append(best_candidate)
        curr_A = np.concatenate(vectors_used)
        out, _, _, _ = np.linalg.lstsq(curr_A, b)
        residual = b - curr_A.dot(out)
    
    return out