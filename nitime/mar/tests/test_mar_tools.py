import numpy as np
import nose.tools as nt
from nitime.mar.mar_tools import lwr, lwr_alternate

def test_lwr_alternate():
    "test solution of lwr recursion"

    for trial in xrange(3):
        nc = np.random.randint(2, high=10)
        P = np.random.randint(2, high=6)
        # nc is channels, P is lags (order)
        r = np.random.randn(P+1,nc,nc)
        r[0] = np.dot(r[0], r[0].T) # force r0 to be symmetric

        a, Va = lwr_alternate(r)
        # Verify the "orthogonality" principle of the mAR system
        # Set up a system in blocks to compute, for each k
        #   sum_{i=1}^{P} A(i)R(-k+i) = -R(-k)  k > 0
        # = sum_{i=1}^{P} (R(-k+i)^T A^T(i))^T = -R(-k) = -R(k)^T
        # = sum_{i=1}^{P} R(k-i)A.T(i) = -R(k)
        rmat = np.zeros((nc*P, nc*P))
        for k in xrange(1,P+1):
            for i in xrange(1,P+1):
                im = k-i
                if im < 0:
                    r1 = r[-im].T
                else:
                    r1 = r[im]
                rmat[(k-1)*nc:k*nc,(i-1)*nc:i*nc] = r1

        rvec = np.zeros((nc*P, nc))
        avec = np.zeros((nc*P, nc))
        for m in xrange(P):
            rvec[m*nc:(m+1)*nc] = -r[m+1]
            avec[m*nc:(m+1)*nc] = a[m].T

        l2_d = np.dot(rmat, avec) - rvec
        l2_d = (l2_d**2).sum()**0.5
        l2_r = (rvec**2).sum()**0.5

        # compute |Ax-b| / |b| metric
        yield nt.assert_almost_equal, l2_d/l2_r, 0


def test_lwr():
    "test solution of lwr recursion"

    for trial in xrange(3):
        nc = np.random.randint(2, high=10)
        P = np.random.randint(2, high=6)
        # nc is channels, P is lags (order)
        r = np.random.randn(P+1,nc,nc)
        r[0] = np.dot(r[0], r[0].T) # force r0 to be symmetric

        a, Va = lwr(r)
        # Verify the "orthogonality" principle of the mAR system
        # Set up a system in blocks to compute, for each k
        #   sum_{i=1}^{P} A(i)R(k-i) = -R(k) k > 0
        # = sum_{i=1}^{P} R(k-i)^T A(i)^T = -R(k)^T
        # = sum_{i=1}^{P} R(i-k)A(i)^T = -R(k)^T
        rmat = np.zeros((nc*P, nc*P))
        for k in xrange(1,P+1):
            for i in xrange(1,P+1):
                im = i-k
                if im < 0:
                    r1 = r[-im].T
                else:
                    r1 = r[im]
                rmat[(k-1)*nc:k*nc,(i-1)*nc:i*nc] = r1

        rvec = np.zeros((nc*P, nc))
        avec = np.zeros((nc*P, nc))
        for m in xrange(P):
            rvec[m*nc:(m+1)*nc] = -r[m+1].T
            avec[m*nc:(m+1)*nc] = a[m].T

        l2_d = np.dot(rmat, avec) - rvec
        l2_d = (l2_d**2).sum()**0.5
        l2_r = (rvec**2).sum()**0.5

        # compute |Ax-b| / |b| metric
        yield nt.assert_almost_equal, l2_d/l2_r, 0
