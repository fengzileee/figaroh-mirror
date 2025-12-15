import numpy as np


class Trajectory:
    def __init__(
        self,
        dt: float,
        joint_configuration: np.ndarray,
        joint_velocity: np.ndarray,
        joint_acceleration: np.ndarray,
        measured_link_side_torque: np.ndarray,
    ):
        self._dt = dt
        self._q = np.array(joint_configuration)
        self._length, self._nq = self._q.shape[0], self._q.shape[1]
        self._dq = np.array(joint_velocity)
        self._nv = self._dq.shape[1]
        self._ddq = np.array(joint_acceleration)
        self._tau = np.array(measured_link_side_torque)
        self._check_dimensions()

    def __len__(self):
        return self._length

    @property
    def configuration(self):
        return self._q

    @property
    def velocity(self):
        return self._dq

    @property
    def acceleration(self):
        return self._ddq

    @property
    def measured_torque(self):
        return self._tau

    def _check_dimensions(self):
        assert (
            self._q.shape[0] == self._tau.shape[0]
        ), "configuration and torque must have the same number of time steps"
        assert (
            self._q.shape[0] == self._dq.shape[0]
        ), "configuration and velocity must have the same number of time steps"
        assert (
            self._q.shape[0] == self._ddq.shape[0]
        ), "configuration and acceleration must have the same number of time steps"
        assert (
            self._dq.shape[1] == self._tau.shape[1]
        ), "velocity and torque must have the same number of joints"
        assert (
            self._dq.shape[1] == self._ddq.shape[1]
        ), "velocity and acceleration must have the same number of joints"

    def random_like(self):
        q = np.random.uniform(low=-6, high=6, size=(len(self), self._nq))
        dq = np.random.uniform(low=-6, high=6, size=(len(self), self._nv))
        ddq = np.random.uniform(low=-6, high=6, size=(len(self), self._nv))
        tau = np.random.uniform(low=-6, high=6, size=(len(self), self._nv))
        return Trajectory(self._dt, q, dq, ddq, tau)
