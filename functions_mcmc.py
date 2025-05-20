import numpy as np
import matplotlib.pyplot as plt


######### AUTOCORRELATION FUNCTIONS

# from https://emcee.readthedocs.io/en/stable/tutorials/autocorr/


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def plot_autocorr(chains, labels=[], show=True):


    nit, nwalkers, ndim = np.shape(chains)

    if len(labels)!=ndim:
        labels=['par'+str(i+1) for i in range(ndim)]

    
    fig=plt.figure(figsize=(10,1*(ndim)))

    axs=[]
    for ip in range(ndim):
        ax=fig.add_subplot(ndim,1,ip+1)
        axs.append(ax)

    for ip in range(ndim):
        axi=axs[ip]

        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chains.shape[1]), 10)).astype(int)
        new = np.empty(len(N))
        for i, n in enumerate(N):
            
             new[i] = autocorr_new(chains[:, :n, ip])

        axi.loglog(N, new, "o-", label="new")
        if ip==ndim-1:
            axi.set_xlabel("number of samples, $N$")
        axi.set_ylabel(r"$\tau$ "+labels[ip])


    plt.tight_layout()
    plt.subplots_adjust(hspace=0, bottom=0.02, top=0.98, right=0.99, left=0.05)
    plt.savefig('autocorr.png')
    if show:
        plt.show()
