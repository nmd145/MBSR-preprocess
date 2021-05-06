# Multi-channel Wiener Filter for EEG artefact removal
# Python implementation based on the original MATLAB toolbox
# https://github.com/exporl/mwf-artifact-removal

import numpy as np
from scipy import linalg

class MWF(object):

    """
    A MWF class that will remove artefacts from EEG signal given the mask of artefact positions

    Attributes:
        n_channels (int): Number of channels
        delay (int): Number of time lags
        rank (str): create MWF filter with one of the following:
            - 'full': all eigenvalues are retained
            - 'poseig': only positive eigenvalues are retained (default)
            - 'first': only k largest eigenvalues are retained
            - 'pct': only k% largest eigenvalues are retained
        mu (float): noise weighting factor
    """

    def __init__(self, delay=0, rank="poseig", mu=1):
        super(MWF, self).__init__()
        # self.n_channels = n_channels
        self.delay = delay
        self.rank = rank
        self.mu = mu

    def ensure_symmetry(self, X):
        """
        Force matrix X to be summetric by averaging with its transpose.

        Args:
            X (np.ndarray): Input 2D numpy array

        Returns:
            (np.ndarray): 2D Numpy array
        """
        if not np.allclose(X, X.T, rtol=1e-05, atol=1e-08):
            X = (X.T + X) / 2
        return X

    def check_dimensions(self, X):
        """
        Validate the dimensions of EEG signal

        Args:
            X (TYPE): Description
        """
        assert len(X.shape) == 2

    def delay_data(self, X, k=0):
        """Create time-delayed matrix stacked with original one

        Args:
            X (np.ndarray): Input 2D numpy array (n_channels x n_samples)
            k (int): maximum number of lags to include
        """
        # delayed = np.roll(X, k, axis=1)
        # delayed[:, :k] = 0
        # return delayed
        if k == 0:
            return X, X.shape[0]
        m = X.shape[0]  # number of channels
        m_s = (k + 1) * m
        # X_s = X.copy()
        X_s = np.zeros((m_s, X.shape[1]))
        X_s[:m, :] = X
        # Horizontally stack lagged versions of the orig data
        for t in range(1, k + 1):
            X_shift = np.roll(X, t, axis=1)
            X_shift[:, :t] = 0
            # X_s = np.concatenate((X_s, X_shift), axis=0)
            X_s[t * m : (t + 1) * m, :] = X_shift

        return X_s, m_s

    def fit(self, X, mask, verbose=False):
        """
        Compute multi-channel Wiener filter (MWF) based on
        generalised eigenvalue decomposition (GEVD) for EEG artefact removal.
        The filter is trained on the raw EEG data using the provifed artefact mask.

        Args:
            X (np.ndarray): raw EEG matrix (n_channels x n_samples)
            mask (np.ndarray): boolean mask of artifacts (1 x n_samples)

        Returns:
            TYPE: Description
        """
        # Create time-lagged data
        X, m_s = self.delay_data(X, self.delay)
        if verbose:
            print("Creating stacked delay data... Done.")
            print(f"Data shape: {X.shape}")

        # Calculate covariance matrices
        R_yy = np.cov(X[:, mask == 1])
        R_nn = np.cov(X[:, mask == 0])
        if verbose:
            print("Creating covariance matrices... Done.")

        # Ensure symmetry
        R_yy = self.ensure_symmetry(R_yy)
        R_nn = self.ensure_symmetry(R_nn)
        if verbose:
            print("Ensuring symmetry... Done.")

        # GEVD
        eig_val, eig_vec = linalg.eig(R_yy, R_nn)
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        assert np.allclose(
            np.dot(R_yy, eig_vec), np.dot(np.dot(R_nn, eig_vec), eig_val)
        )
        # Convert to diagonal
        # print(f"original eigenvalues: {eig_val}")
        eig_val = np.diag(eig_val)
        # print(f"diag matrix with eigvalues on diagonal: {eig_val}")
        Lambda_y = np.linalg.multi_dot([eig_vec.T, R_yy, eig_vec])
        Lambda_n = np.linalg.multi_dot([eig_vec.T, R_nn, eig_vec])
        Delta = Lambda_y - Lambda_n
        if verbose:
            print(f"Delta shape: {Delta.shape}")

        # Eigenvectors are assumed to be scaled such that Lambda_n is approx. identity
        diffs = np.abs(Lambda_n - np.eye(m_s))
        if (diffs > 1e-2).any():
            print("warning...")
            warnings.warn(
                """
                Generalised eigenvectors are not scaled as assumed: results may be inacurrate.
                This is likely caused by (almost) rank deficient covariance matrices.
                Make sure that the EEG has full rank and that the mask provides enough
                clean/artifact EEG samples for covariance matrix estimation.
            """
            )
        # Set filter rank
        if self.rank == "full":  # full rank MWF
            rank_w = m_s
            message = f"Keeping all eigen values... Rank = {rank_w}"
        elif self.rank == "poseig":
            n_reject = np.sum(np.diag(Delta) < 0)
            rank_w = m_s - n_reject
            message = f"Rejecting {n_reject} negative eigenvalues. Total: {m_s}... Rank = {rank_w}"
        if verbose:
            print(message)
            print(Delta.shape)
            print(eig_val.shape)
            print(eig_vec.shape)
        # Create filter of rank specified above
        #         Delta[rank_w * (m_s + 1) + 1 : m_s+1 : m_s*m_s]
        #         print((eig_val + (self.mu - 1) * np.eye(m_s)).shape)
        Delta[rank_w:, rank_w * (m_s + 1) : m_s * m_s : m_s + 1] = 0
        # print(rank_w * (m_s + 1), m_s + 1, m_s * m_s)
        temp1 = eig_val + (self.mu - 1) * np.eye(m_s)
        temp2 = np.linalg.lstsq(temp1, eig_vec)[0]  # equiv to eig_vec / temp1
        temp3 = np.dot(temp2, Delta)
        temp4 = np.linalg.lstsq(eig_vec, temp3)[0]  # equiv to temp3 / eig_vec
        #         W = eig_vec / (eig_val + (self.mu - 1) * np.eye(m_s)) * Delta / V
        self.W = temp4
        self.Lambda = eig_val
        print(f"W shape: {self.W.shape}")
        print(f"lambda shape: {self.Lambda.shape}")
        return self

    def transform(self, X):
        """
        Apply a precomputed Multi-channel Wiener filter W on a multi-channel EEG signal
        Args:
            X (np.ndarray): raw EEG matrix (n_channels x n_samples)

        Returns:
            TYPE: Description
        """
        m, t = X.shape
        m_s = self.W.shape[0]
        print(m, m_s)
        tau = (m_s - m) / (2 * m)
        print(tau)
        if tau % 1 != 0:
            raise ValueError(
                "The given filter is not compatible with the input EEG signal."
            )
        # Subtract mean from data
        channel_means = X.mean(axis=1)
        X = X - np.repeat(channel_means.reshape(-1, 1), t, axis=1)

        # Reapply time lags
        tau = int(tau)
        X_s, _ = self.delay_data(X, tau)

        # print(self.W.shape)
        # Compute artefact estimate for original channels of X
        # orig_chans = np.s_[tau * m + 1 : (tau + 1) * m]
        # print(orig_chans)
        print(self.W[:, tau * m : (tau + 1) * m].T.shape)
        self.d = np.dot(self.W[:, tau * m : (tau + 1) * m].T, X_s)

        # Subtract artefact estimate from data
        n = X - self.d

        # Add mean back to filtered EEG
        n = n + np.repeat(channel_means.reshape(-1, 1), t, axis=1)

        return n
