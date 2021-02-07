
"""
Solution stub for Question 1 (Rejection Sampling).

Fill in the implementation of the `rejection_sampled_expectation` function.

Every `Distribution` object has three methods:
- pdf(x): returns the density (possibly unnormalized) of the distribution at x
- max_pdf(): returns an upper bound on the return value of pdf
- sample(): returns a single value sampled from the distribution.
            not implemented for unnormalized distributions.

See `monte_carlo_expectation` function for an example of using `Distribution`
objects to compute expectations of functions.

You can generate uniformly-distributed values on [0,1] by calling uniform().
"""

from __future__ import print_function
from numpy import mean, isclose  # mean(array_or_list) = average of values in array_or_list
from rs_utils import Uniform, Gaussian, TruncatedGaussian

uniform = Uniform(0.0, 1.0)

# Some target / proposal distributions
norm25 = Gaussian(2.5, 1.0)
U02 = Uniform(0.0, 2.0)
U33 = Uniform(-3, 3)
half_trunc = TruncatedGaussian(0.0, 1.0, a=0.0, b=1.0)
trunc = TruncatedGaussian(1.75, 0.5, a=0.75, b=2.75)
untrunc = Gaussian(1.75, 0.5)

def identity(x):
    return x

def monte_carlo_expectation(target, f=identity, num_samples=2000):
    """
    Compute the expected value of `f(X)` for a random variable `X` with
    distribution `target`, by drawing `num_samples` samples from `target`.

    Args:
        target:
          A `Distribution` object that provides `pdf`, `max_pdf`, and
          `sample` methods, representing the target distribution.
        f:
          A single-argument function to compute expectation for; defaults to
          the `identity` function.
        num_samples:
          The number of samples from `target` to use; defaults to 2000.

    Returns:
        The estimated expected value of `f(X)`.
    """
    samples = []
    for n in range(num_samples):
        v = target.sample() # sample a single value from the distribution
        samples.append(f(v))

    # Average value of the sample is an estimate of the expected value
    return mean(samples)


def rejection_sampled_expectation(target, proposal, f=identity, num_samples=2000):
    """
    Compute the expected value of `f(X)`, for a random variable `X` with
    distribution `target` by rejection sampling.  Samples will be drawn from
    `proposal` and rejected based on the UNNORMALIZED densities provided by
    `target`.

    Args:
        target:
          A `Distribution` object that provides `pdf` and `max_pdf`
          methods, representing the target distribution.
        proposal:
          A `Distribution` object that provides `pdf`, `max_pdf`, and
          `sample` methods, representing the proposal distribution.
        f:
          A single-argument function to compute expectation for; defaults to
          the `identity` function.
        num_samples:
          The number of *unrejected* samples to use; defaults to 2000.

    Returns:
        The estimated expected value of `f(X)`.
    """
    #TODO: Add implementation here
    return 0.0

def main():
    print("Basic tests...")

    # Usage examples
    assert 0.0 <= U02.sample() <= 2.0
    assert U02.pdf(1.4) == 0.5
    assert U02.pdf(7.0) == 0.0
    assert U02.max_pdf() == 0.5
    
    # Check that monte_carlo_expectation works
    assert isclose(1.0, monte_carlo_expectation(U02), atol=0.1)
    assert isclose(2.5, monte_carlo_expectation(norm25), atol=0.1)
    assert isclose(1.0, monte_carlo_expectation(norm25, lambda x: (x - 2.5)**2), atol=0.1)

    print("ok")

    # Test rejection sampling
    print("test target=U02, proposal=U02...")
    assert isclose(6.0, rejection_sampled_expectation(U02, U02, f=lambda x: 5+x), atol=0.1)
    print("ok")

    print("test target=norm25, proposal=norm25...")
    assert isclose(2.5, rejection_sampled_expectation(norm25, norm25), atol=0.1)
    assert isclose(1.0, rejection_sampled_expectation(norm25, norm25, lambda x: (x - 2.5)**2), atol=0.1)
    print("ok")

    print("test target=half_trunc, proposal=U02...")
    assert isclose(0.46, rejection_sampled_expectation(half_trunc, U02), atol=0.1)
    print("ok")

    print("test target=trunc, proposal=U33...")
    assert isclose(1.75, rejection_sampled_expectation(trunc, U33), atol=0.1)
    assert isclose(3.5, rejection_sampled_expectation(trunc, U33, f=lambda x: 2*x), atol=0.1)
    print("ok")
    
    print("test target=trunc, proposal=untrunc...")
    assert isclose(1.75, rejection_sampled_expectation(trunc, untrunc), atol=0.1)
    assert isclose(3.5, rejection_sampled_expectation(trunc, untrunc, f=lambda x: 2*x), atol=0.1)
    print("ok")
    
if __name__ == '__main__':
    main()
