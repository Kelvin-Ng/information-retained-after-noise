import numpy as np
from scipy import stats
import scipy.integrate as integrate
import sys

def main(dac_bits, num_bits, num_padded_bits, snr_db):
    max_signal_rms = 2 ** dac_bits / 2. / np.sqrt(2) # Assuming sine signal with maximum amplitude
    snr_ratio = 10. ** (snr_db / 20.)
    sigma = max_signal_rms / snr_ratio

    # "Signal range" here means, for one discrete signal `s`, what range in the analog observation space do we consider the observation as signal s.
    # In the basic situation, the "width" is 1 "unit" of amplitude, i.e., if we see an analog signal of [s - 1/2, s + 1/2], we consider that we accurately
    # obtain the signal `s`.
    # However, there are cases where we have paddings in the LSB. For example, if we play a 24-bit sample in a 32-bit DAC, there are 8 bits of paddings.
    # We don't care about the padding part, so we now have a range of [s - 1/2^7, s + 1/2^7].
    signal_range_div_2 = 2 ** (num_padded_bits - 1)

    noise_dist = stats.Normal(mu=0, sigma=sigma) # Assuming noise of normal distribution

    log_of_2 = np.log(2)

    # We assume a model of O = S + N, where O, S, N are random variables representing obervation, signal, and noise respectively.
    # We now calculate the mutual information of S given O, i.e., I(S; O). It means by observing O, which is basically signal polluted by noise,
    # how much information we can still learn about the original signal.

    # The first part is the joint distribution P_O,S, which is 1/2^b * P_N(o - s). We skip the 1/2^b because we need to multiply 2^b later.

    # The second part is log2(P_S|O(s, o) / P_S(s)).

    # log2(P_S(s)) is -b, hence the `+ num_bits` in the formula.

    # P_S|O(s, o) means the probability of S=s given an observation O=o. One can easily calculate this with the formula P_S,O(s, o) / P_O(o).
    # However, P_O will involve a summation over all possible values of S, which is very computation intensive.

    # Another approach, which is used here, is to calculate this directly.
    # A quick thought will gives P_S|O(s, o) = P_N(o - s) (by using the relation of O = S + N).
    # However, this is incorrect for discrete S. P_S is a probabilty while P_N is only a probability density.
    # As discussed earlier, for discrete S, a range of analog observation can still be considered to be an accurate S.
    # So, P_S|O(s, o) is actually the integral of P_N(o - x) over the range x in [s - r/2, s + r/2], where r/2 is `signal_range_div_2`.
    # After some rearragement yields P_S|O(s, o) = integral of P_N(u) over the range u in [o - s - r/2, o - s + r/2].
    # P_S|O(s, o) is inside a log, and the integration is essentially the range CDF. Hence, we use the logcdf function provided by scipy.
    # Putting all these together yields the follow `inner()` function, which is the function to be integrated over `o`.

    def inner(o, s):
        return noise_dist.pdf(o - s) * (noise_dist.logcdf(o - s - signal_range_div_2, o - s + signal_range_div_2) / log_of_2 + num_bits)

    # It does not look like there is an analytical formula for the integral of `inner()` over `o`, so we use numerical integral.
    # In the raw formula, we need to do a summation over `s` of the integral over `o`.
    # However, we know that the result of the integral is a constant across `s`.
    # So, we only need to do one integral using an arbitrary `s` and multiply it by the range of the summation, which is 2^b.
    # However, due to numerical inaccuracy, `s` with large absolute values will give bad results, so we choose s = 0.
    # Multiplying it by the range of `s` is already inherent in the `inner()` function (by not multiplying 1/2^b).
    s = 0
    res = integrate.quad(lambda o: inner(o, s), -np.inf, np.inf)

    print('Number of bits retained:', res[0])
    print('Error bound:', res[1])

if __name__ == '__main__':
    dac_bits = int(sys.argv[1]) # Bit depth of the DAC
    num_bits = int(sys.argv[2]) # Number of bits in the source
    num_padded_bits = int(sys.argv[3]) # Number of padded bits in the LSB (not included in the `num_bits`)
    snr_db = float(sys.argv[4]) # SNR in dB

    main(dac_bits, num_bits, num_padded_bits, snr_db)

