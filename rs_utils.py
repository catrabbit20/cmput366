"""
Utility classes for rejectionSampling.
"""
import scipy.stats

class Distribution(object):
    def pdf(self, x):
        raise NotImplementedError

    def max_pdf(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    

class Uniform(Distribution):
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b
        self._rv = scipy.stats.uniform(loc=a, scale=b-a)

    def pdf(self, x):
        return self._rv.pdf(x)

    def max_pdf(self):
        return self._rv.pdf((self.b - self.a) / 2)

    def sample(self):
        return self._rv.rvs()

class Gaussian(Distribution):
    def __init__(self, mean=0.0, sd=1.0):
        self.mean = mean
        self.sd = sd
        self._rv = scipy.stats.norm(loc=mean, scale=sd)

    def pdf(self, x):
        return self._rv.pdf(x)

    def max_pdf(self):
        return self._rv.pdf(self.mean)

    def sample(self):
        return self._rv.rvs()

class TruncatedGaussian(Distribution):
    def __init__(self, mean=0.0, sd=1.0, a=0.0, b=1.0, k=1.25):
        if not (a <= mean <= b):
            raise ValueError("mean %f is outside truncation range [%f, %f]" % (mean, a, b))
        if k <= 0.0:
            raise ValueError("scaling factor k must be positive")
        self.a = a
        self.b = b
        self.k = k
        self.mean = mean
        self.sd = sd
        self._rv = scipy.stats.norm(loc=mean, scale=sd)

    def pdf(self, x):
        if self.a <= x <= self.b:
            return self._rv.pdf(x) * self.k
        else:
            return 0.0

    def max_pdf(self):
        return self.pdf(self.mean)

    def sample(self):
        raise NotImplementedError
