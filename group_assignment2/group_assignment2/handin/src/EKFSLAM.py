from cmath import sin
from typing import Tuple
import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field
from scipy.linalg import block_diag
import scipy.linalg as la
from utils import rotmat2d
from JCBB import JCBB
import utils
import solution.EKFSLAM


@dataclass
class EKFSLAM:
    Q: ndarray
    R: ndarray
    do_asso: bool
    alphas: 'ndarray[2]' = field(default=np.array([0.001, 0.0001]))
    sensor_offset: 'ndarray[2]' = field(default=np.zeros(2))

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Add the odometr to the robot state x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray, shape = (3,)
            the predicted state
        """
        xpred = np.empty(3, dtype=float)
        xpred[0] = x[0] + u[0] *np.cos(x[2]) - u[1]*np.sin(x[2])
        xpred[1] = x[1] + u[0] *np.sin(x[2]) + u[1]*np.cos(x[2])
        xpred[2] = x[2] + u[2]
        xpred[2] = utils.wrapToPi(xpred[2])
        return xpred


    def Fx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to x.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. x.
        """
        Fx = np.block([
            [1, 0, -u[0]*np.sin(x[2]) - u[1]*np.cos(x[2])],
            [0, 1, u[0]*np.cos(x[2]) - u[1]*np.sin(x[2])],
            [0, 0, 1]
        ])
        return Fx


    def Fu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate the Jacobian of f with respect to u.

        Parameters
        ----------
        x : np.ndarray, shape=(3,)
            the robot state
        u : np.ndarray, shape=(3,)
            the odometry

        Returns
        -------
        np.ndarray
            The Jacobian of f wrt. u.
        """
        Fu = np.block([
            [np.cos(x[2]), -np.sin(x[2]), 0],
            [np.sin(x[2]), np.cos(x[2]), 0],
            [0, 0, 1]
        ])
        return Fu



    def predict(
        self, eta: np.ndarray, P: np.ndarray, z_odo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the robot state using the zOdo as odometry the corresponding state&map covariance.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z_odo : np.ndarray, shape=(3,)
            the measured odometry

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes= (3 + 2*#landmarks,), (3 + 2*#landmarks,)*2
            predicted mean and covariance of eta.
        """
        M = np.zeros((eta.shape[0], 3 ), dtype=float)
        M[:3, :] = np.eye(3)
        F_x = self.Fx(eta[:3], z_odo)
        F = np.block([[F_x, np.zeros((F_x.shape[0], eta.shape[0]-3))],
                      [np.zeros((eta.shape[0]-3, F_x.shape[1])), np.eye(eta.shape[0]-3)]])
        
        xpred = self.f(eta[:3], z_odo)
        etapred = eta.copy()
        etapred[:3] = xpred

        P[:, :] = F @ P @ F.T + M @ self.Q @ M.T
        return etapred, P


    def h(self, eta: np.ndarray) -> np.ndarray:
        """Predict all the landmark positions in sensor frame.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks,)
            The landmarks in the sensor frame.
        """

        zpred = []
        offset = rotmat2d(eta[2]) @ self.sensor_offset
        for i in range(3, len(eta), 2):
            zpred_0 = np.sqrt((eta[i:i+1]-eta[0]-offset[0])**2+(eta[i+1:i+2]-eta[1]-offset[1])**2)
            zpred_1 = rotmat2d(-eta[2])@(eta[i:i+2]-eta[0:2]-offset)
            zpred_1 = np.arctan2(zpred_1[1], zpred_1[0])
            zpred.append(np.hstack((zpred_0, [zpred_1])))
        zpred = np.hstack(zpred) 
        return zpred

    

    def h_jac(self, eta: np.ndarray) -> np.ndarray:
        """Calculate the jacobian of h.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2 * #landmarks,)
            The robot state and landmarks stacked.

        Returns
        -------
        np.ndarray, shape=(2 * #landmarks, 3 + 2 * #landmarks)
            the jacobian of h wrt. eta.
        """

        m = eta[3:].reshape((-1, 2))   
        numM = m.shape[0]

        H = np.zeros((2 * numM, 3 + 2 * numM))
        offset = rotmat2d(eta[2]) @ self.sensor_offset

        for i in range(numM):   
            z_c = m[i] - eta[:2] - offset

            z_b_0 = (z_c.T/np.linalg.norm(z_c))@ np.column_stack((-np.eye(2), -rotmat2d(np.pi/2) @ (m[i]-eta[:2])))
            z_b_1 = z_c.T@rotmat2d(np.pi/2).T/np.linalg.norm(z_c)**2 @ np.column_stack((-np.eye(2), -rotmat2d(np.pi/2) @ (m[i]-eta[:2])))
            
            H_x = np.vstack([z_b_0, z_b_1])  

            z_m_0 = z_c.T/np.linalg.norm(z_c)
            z_m_1 = z_c.T@rotmat2d(np.pi/2).T/np.linalg.norm(z_c)**2
            H_m = np.vstack([z_m_0, z_m_1])

            row = 2*i
            col = 3 + 2*i
            H[row:row+2, :3] = H_x
            H[row:row+2, col:col+2] = H_m

        return H

        

    def add_landmarks(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new landmarks, their covariances and add them to the state.

        Parameters
        ----------
        eta : np.ndarray, shape=(3 + 2*#landmarks,)
            the robot state and map concatenated
        P : np.ndarray, shape=(3 + 2*#landmarks,)*2
            the covariance of eta
        z : np.ndarray, shape(2 * #newlandmarks,)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray], shapes=(3 + 2*(#landmarks + #newlandmarks,), (3 + 2*(#landmarks + #newlandmarks,)*2
            eta with new landmarks appended, and its covariance
        """
        n = P.shape[0]

        numLmk = z.shape[0] // 2
        lmnew = np.empty_like(z)  

        Gx = np.empty((2 * numLmk, 3))
        Rall = np.zeros((2 * numLmk, 2 * numLmk))

        rho = eta[:2]
        psi = eta[2]

        sensor_offset_world = rotmat2d(psi) @ self.sensor_offset
        sensor_offset_world_der = rotmat2d(psi + np.pi / 2) @ self.sensor_offset

        for j in range(numLmk):
            ind = 2 * j
            inds = slice(ind, ind + 2)
            r, phi = z[inds]

            s_body = np.array([r * np.cos(phi), r * np.sin(phi)])           
            s_world = rotmat2d(psi) @ s_body                                         
            lm = rho + s_world + sensor_offset_world                        
            lmnew[inds] = lm

            Gx[inds, :2] = np.eye(2)
            dpsi = r * np.array([-np.sin(phi + psi), np.cos(phi + psi)]) + sensor_offset_world_der
            Gx[inds, 2] = dpsi                                              

            
            rot = rotmat2d(phi + psi)
            Gz = rot @ np.diag([1.0, r])


            Rall[inds, inds] = Gz @ self.R @ Gz.T

        
        etaadded = np.concatenate([eta, lmnew])

        
        Padded = np.zeros((n + 2 * numLmk, n + 2 * numLmk))
        Padded[:n, :n] = P

        Px_all = P[:3, :]          
        P_allx = P[:, :3]          
        P_xx  = P[:3, :3]          

        
        Padded[n:, :n] = Gx @ Px_all
        Padded[:n, n:] = P_allx @ Gx.T 

        Padded[n:, n:] = Gx @ P_xx @ Gx.T + Rall


        return etaadded, Padded

       
    def associate(
        self, z: np.ndarray, zpred: np.ndarray, H: np.ndarray, S: np.ndarray,
    ):  # -> Tuple[*((np.ndarray,) * 5)]:
        """Associate landmarks and measurements, and extract correct matrices for these.

        Parameters
        ----------
        z : np.ndarray,
            The measurements all in one vector
        zpred : np.ndarray
            Predicted measurements in one vector
        H : np.ndarray
            The measurement Jacobian matrix related to zpred
        S : np.ndarray
            The innovation covariance related to zpred

        Returns
        -------
        Tuple[*((np.ndarray,) * 5)]
            The extracted measurements, the corresponding zpred, H, S and the associations.

        Note
        ----
        See the associations are calculated  using JCBB. See this function for documentation
        of the returned association and the association procedure.
        """
        if self.do_asso:
            # Associate
            a = JCBB(z, zpred, S, self.alphas[0], self.alphas[1])

            # Extract associated measurements
            zinds = np.empty_like(z, dtype=bool)
            zinds[::2] = a > -1  # -1 means no association
            zinds[1::2] = zinds[::2]
            zass = z[zinds]

            # extract and rearange predicted measurements and cov
            zbarinds = np.empty_like(zass, dtype=int)
            zbarinds[::2] = 2 * a[a > -1]
            zbarinds[1::2] = 2 * a[a > -1] + 1

            zpredass = zpred[zbarinds]
            Sass = S[zbarinds][:, zbarinds]
            Hass = H[zbarinds]

            assert zpredass.shape == zass.shape
            assert Sass.shape == zpredass.shape * 2
            assert Hass.shape[0] == zpredass.shape[0]

            return zass, zpredass, Hass, Sass, a
        else:
            # should one do something her
            pass

    def update(
        self, eta: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Update eta and P with z, associating landmarks and adding new ones.

        Parameters
        ----------
        eta : np.ndarray
            the robot state and map concatenated
        P : np.ndarray
            the covariance of eta
        z : np.ndarray, shape=(#detections, 2)
            A set of measurements to create landmarks for

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]
            updated eta, updated P, NIS, and the associations
        """

        numLmk = (eta.size - 3) // 2

        if numLmk > 0:
            # Predict measurements and Jacobian
            zpred = self.h(eta)                 # shape (2*numLmk,)
            H = self.h_jac(eta) 

            # Here you can use simply np.kron (a bit slow) to form the big (very big in VP after a while) R,
            # or be smart with indexing and broadcasting (3d indexing into 2d mat) realizing you are adding the same R on all diagonals
            S = H @ P @ H.T + np.kron(np.eye(numLmk), self.R)
            assert S.shape == zpred.shape * 2, "EKFSLAM.update: wrong shape on either S or zpred"

            # Flatten incoming detections
            z = z.ravel()

            # Data association → keep only associated pieces
            za, zpred, Ha, Sa, a = self.associate(z, zpred, H, S)

            # If nothing associated, skip update
            if za.shape[0] == 0:
                etaupd = eta
                Pupd = P
                NIS = 1.0  # placeholder for consistency analysis
            else:
                # Innovation
                v = za.ravel() - zpred                # (2*k,)
                v[1::2] = utils.wrapToPi(v[1::2])

                # Kalman gain (solve Sa * X = (Ha P)^T)
                # W = P Ha^T Sa^{-1}
                W = P @ Ha.T @ np.linalg.solve(Sa, np.eye(Sa.shape[0]))

                # Mean update
                etaupd = eta + W @ v

                # Joseph-form covariance: P' = (I-KH)P(I-KH)^T + K R_a K^T
                jo = -W @ Ha
                jo[np.diag_indices(jo.shape[0])] += 1.0

                # Build R_a = blkdiag(self.R, ..., self.R) for the k associated measurements
                k = za.size // 2
                R_a = np.kron(np.eye(k), self.R)

                Pupd = jo @ P @ jo.T + W @ R_a @ W.T

                # NIS = v^T S^{-1} v
                NIS = float(v.T @ np.linalg.solve(Sa, v))

                assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd not symmetric"
                assert np.all(np.linalg.eigvals(Pupd) > 0), "EKFSLAM.update: Pupd not positive definite"

        else:
            # No landmarks yet → all detections will be new
            a = np.full(z.shape[0], -1)
            z = z.flatten()
            NIS = 1.0
            etaupd = eta
            Pupd = P

        # Add any new landmarks (a == -1)
        if self.do_asso:
            is_new_lmk = a == -1
            if np.any(is_new_lmk):
                z_new_inds = np.empty_like(z, dtype=bool)
                z_new_inds[::2] = is_new_lmk
                z_new_inds[1::2] = is_new_lmk
                z_new = z[z_new_inds]
                etaupd, Pupd = self.add_landmarks(etaupd, Pupd, z_new)

        assert np.allclose(Pupd, Pupd.T), "EKFSLAM.update: Pupd must be symmetric"
        assert np.all(np.linalg.eigvals(Pupd) >= 0), "EKFSLAM.update: Pupd must be PSD"

        return etaupd, Pupd, NIS, a

 

    @classmethod
    def NEESes(cls, x: np.ndarray, P: np.ndarray, x_gt: np.ndarray,) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates
        Args:
            x (np.ndarray): The estimate
            P (np.ndarray): The state covariance
            x_gt (np.ndarray): The ground truth
        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties
        Returns:
            np.ndarray: NEES for [all, position, heading], shape (3,)
        """

        assert x.shape == (3,), f"EKFSLAM.NEES: x shape incorrect {x.shape}"
        assert P.shape == (3, 3), f"EKFSLAM.NEES: P shape incorrect {P.shape}"
        assert x_gt.shape == (
            3,), f"EKFSLAM.NEES: x_gt shape incorrect {x_gt.shape}"

        d_x = x - x_gt
        d_x[2] = utils.wrapToPi(d_x[2])
        assert (
            -np.pi <= d_x[2] <= np.pi
        ), "EKFSLAM.NEES: error heading must be between (-pi, pi)"

        d_p = d_x[0:2]
        P_p = P[0:2, 0:2]
        assert d_p.shape == (2,), "EKFSLAM.NEES: d_p must be 2 long"
        d_heading = d_x[2]  # Note: scalar
        assert np.ndim(
            d_heading) == 0, "EKFSLAM.NEES: d_heading must be scalar"
        P_heading = P[2, 2]  # Note: scalar
        assert np.ndim(
            P_heading) == 0, "EKFSLAM.NEES: P_heading must be scalar"

        # NB: Needs to handle both vectors and scalars! Additionally, must handle division by zero
        NEES_all = d_x @ (np.linalg.solve(P, d_x))
        NEES_pos = d_p @ (np.linalg.solve(P_p, d_p))
        try:
            NEES_heading = d_heading ** 2 / P_heading
        except ZeroDivisionError:
            NEES_heading = 1.0 

        NEESes = np.array([NEES_all, NEES_pos, NEES_heading])
        NEESes[np.isnan(NEESes)] = 1.0 

        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes
