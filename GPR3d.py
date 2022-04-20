import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
# from sklearn import 
import sklearn

print ("GPR3d Packages loaded.")

class gpr3d(object):
    def __init__(self, _name):
        self.name = _name
        # BY DEFAULT, DONT DO EPSILON RUN-UP
        self.DO_EPSRU = False
        print ("[%s] GAUSSIAN PROCESS REGRESSION 3D" % (self.name))

    """
        SET ANCHOR POINTS & HYPERPARAMETERS & VELOCITY
    """
    def set_data(self, _px, _hyp_mean, _hyp_var, _DO_EPSRU=False, _teps=0.01):
        # SET POSITIONS (ANCHORS) TO FOLLOW
        self.px = _px
        # NUMBER OF ANCHORS
        self.nx = self.px.shape[0]
        # DIMENSION OF POSITION
        self.dim = self.px.shape[1]
        # SET HYPERPARAMETERS (FOR BOTH MEAN AND VAR)
        self.hyp_mean  = _hyp_mean
        self.hyp_var   = _hyp_var
        # SET TIME INDICES
        self._get_timeidx()
        # EPSILON RUN-UP
        if _DO_EPSRU:
            self._do_epsrunup(_teps)

    """
        KERNEL FUNCTION
    """
    def kernel_se(self, _X, _Y, _hyp={'gain':1, 'len':1}):
        hyp_gain = float(_hyp['gain'])
        hyp_len  = 1/float(_hyp['len'])
        pairwise_dists = cdist(_X, _Y, 'euclidean')
        K = hyp_gain*np.exp(-pairwise_dists ** 2 / hyp_len)
        return K

    """
        COMPUTE TIME INDICES FOR SELF.PX
            (CALLED INSIDE 'SET_DATA')
    """
    def _get_timeidx(self):
        # TIME INDICES FOR TRAINING ANCHOR POINTS
        self.tx  = np.zeros((self.nx, 1))
        self.sum_dist = 0.0
        self.max_dist = 0.0
        for i in range(self.nx-1):
            prevx    = self.px[i, :]
            currx    = self.px[i+1, :]
            dist     = np.linalg.norm(currx-prevx)
            self.tx[i+1]  = self.tx[i]+dist
            # COMPUTE THE TOTAL DISTANCE
            self.sum_dist = self.sum_dist + dist
            # COMPUTE THE MAXIMUM DISTANCE BETWEEN POINTS
            if dist > self.max_dist:
                self.max_dist = dist

        # NORMALIZE TIME INDICES TO BE WITHIN 0~1
        self.tx = self.tx / self.tx[-1]

    """
        ADD EPSILON RUN-UP TO BOTH 'TX_RU' AND 'PX_RU'
            (CALLED INSIDE 'SET_DATA')
    """
    def _do_epsrunup(self, _teps=0.01):
        # NOW WE ARE DOING EPSILON RUN-UP
        self.DO_EPSRU   = True
        self.teps       = _teps
        self.tx_ru = np.insert(self.tx, 1, self.teps, 0)
        self.tx_ru = np.insert(self.tx_ru, -1, 1-self.teps, 0)

        # EPS-RUNUP OF START
        diff = (self.px[1,:]-self.px[0,:])
        uvec = diff / np.linalg.norm(diff)
        peru = self.px[0,:] + uvec*self.sum_dist*_teps
        self.px_ru = np.insert(self.px, 1, peru, 0)

        # EPS-RUNUP OF END
        diff = (self.px[-1,:]-self.px[-2,:])
        uvec = diff / np.linalg.norm(diff)
        peru = self.px[-1,:] - uvec*self.sum_dist*_teps
        self.px_ru = np.insert(self.px_ru, -1, peru, 0)

    """
        COMPUTE MEAN AND VARAINCE PATHS
    """
    def compute_grp(self, _vel):
        # NUMBER OF POINTS
        self.nz = (int)(self.sum_dist/_vel)
        # TIME INDICES FOR INTERPOLATED PATHS (Z)
        self.tz = np.linspace(0, 1, self.nz).reshape(-1, 1)
        # GET TIME INDICES AND TRAJECTORY FOR TRAINING
        if self.DO_EPSRU:
            self.tx_used = self.tx_ru
            self.px_used = self.px_ru
        else:
            self.tx_used = self.tx
            self.px_used = self.px
        # COMPUTE MEAN PATH
        k_zz = self.kernel_se(self.tz, self.tz, self.hyp_mean)
        self.k_zx_mean = self.kernel_se(self.tz, self.tx_used, self.hyp_mean)
        self.k_xx_mean = self.kernel_se(self.tx_used, self.tx_used, self.hyp_mean)\
                + self.hyp_mean['noise']*np.eye(self.tx_used.shape[0])
        px_used_mean = self.px_used.mean(axis=0)
        self.muz = np.matmul(self.k_zx_mean,
                np.linalg.solve(self.k_xx_mean, self.px_used-px_used_mean))+px_used_mean

        # COMPUATE VARAINCE


        print ("[%s] THE LENGTH OF A INTERPOLATED TRAJ IS [%d]" % (self.name, self.nz))