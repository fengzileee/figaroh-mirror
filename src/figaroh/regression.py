from typing import Tuple, List, Sequence
import re
import logging
from copy import deepcopy

import numpy as np
from pinocchio.robot_wrapper import RobotWrapper

from figaroh.tools.regressor import build_regressor_basic, get_index_eliminate
from figaroh.tools.qrdecomposition import get_baseParams
from figaroh.trajectory import Trajectory
from figaroh.parameter import Parameter

_params_per_body = (
    "Ixx",
    "Ixy",
    "Ixz",
    "Iyy",
    "Iyz",
    "Izz",
    "mx",
    "my",
    "mz",
    "m",
    "Ia",
    "fv",
    "fs",
    "off",
)


logger = logging.getLogger(__name__)


def get_standard_parameters(model, param):
    """This function prints out the standard inertial parameters defined in urdf model.
    Output: params_std: a dictionary of parameter names and their values"""

    phi = []
    params = []

    params_name = (
        "Ixx",
        "Ixy",
        "Ixz",
        "Iyy",
        "Iyz",
        "Izz",
        "mx",
        "my",
        "mz",
        "m",
    )

    # change order of values in phi['m', 'mx','my','mz','Ixx','Ixy','Iyy','Ixz', 'Iyz','Izz'] - from pinoccchio
    # corresponding to params_name ['Ixx','Ixy','Ixz','Iyy','Iyz','Izz','mx','my','mz','m']

    model_friction = model.friction
    model_damping = model.damping
    for i in range(1, len(model.inertias)):
        P = model.inertias[i].toDynamicParameters()
        P_mod = np.zeros(P.shape[0])
        P_mod[9] = P[0]  # m
        P_mod[8] = P[3]  # mz
        P_mod[7] = P[2]  # my
        P_mod[6] = P[1]  # mx
        P_mod[5] = P[9]  # Izz
        P_mod[4] = P[8]  # Iyz
        P_mod[3] = P[6]  # Iyy
        P_mod[2] = P[7]  # Ixz
        P_mod[1] = P[5]  # Ixy
        P_mod[0] = P[4]  # Ixx
        for j in params_name:
            params.append(j + str(i))
        for k in P_mod:
            phi.append(k)

        params.extend(["Ia" + str(i)])
        params.extend(["fv" + str(i), "fs" + str(i)])
        params.extend(["off" + str(i)])

        if param["has_actuator_inertia"]:
            try:
                phi.extend([param["Ia"][i - 1]])
            except Exception as e:
                print("Warning: ", "has_actuator_inertia_%d" % i, e)
                phi.extend([0])
        else:
            phi.extend([0])
        if param["has_friction"]:
            phi.extend([model_friction[i - 1], model_damping[i - 1]])
        else:
            phi.extend([0, 0])
        if param["has_joint_offset"]:
            try:
                phi.extend([param["off"][i - 1]])
            except Exception as e:
                print("Warning: ", "has_joint_offset_%d" % i, e)
                phi.extend([0])
        else:
            phi.extend([0])

    return params, phi


def extract_linear_relation_between_base_and_standard(
    params_base: List[str], params_standard: List[str]
):
    def parse_expression(expr_str):
        """
        Parses a string like "Izz1 + 1.0*Iyy2 + ..." into {variable: coefficient, ...}.
        """
        expr_clean = re.sub(r"(?<!e)(-)", r"+-", expr_str)
        terms = [t.strip() for t in expr_clean.split("+") if t.strip()]
        coeffs = {}
        for term in terms:
            m = re.match(
                r"([+-]?\s*\d*\.?\d+(?:e[+-]?\d+)?)\s*\*\s*([A-Za-z]+\d+)", term
            )
            if m:
                coeff, var = float(m.group(1).replace(" ", "")), m.group(2).strip()
            else:
                var, coeff = term.replace(" ", ""), 1.0
            coeffs[var] = coeff
        return coeffs

    base_param_size = len(params_base)
    standard_param_size = len(params_standard)

    # base_param = A @ standard_param
    A_auto = np.zeros((base_param_size, standard_param_size))
    for i, expr in enumerate(params_base):
        coeffs = parse_expression(expr)
        for var, coeff in coeffs.items():
            if var in params_standard:
                A_auto[i, params_standard.index(var)] = coeff
            else:
                logger.warn(f"{var} not found in phi_vars")
    return A_auto


class Regression:
    """Regression class for building regressor matrix, paramter, and target torque vector.

    target = regressor * parameter
    """

    def __init__(self, robot: RobotWrapper, trajectory: Trajectory):
        self._robot = robot
        self._trajectory = trajectory
        self._config = {
            "is_joint_torques": True,
            "has_actuator_inertia": False,
            "Ia": None,
            "has_friction": True,
            "has_joint_offset": False,
            "off": None,
            "is_external_wrench": False,
            "force_torque": ["All"],
        }

    def get_target(self) -> np.ndarray:
        """Get the vector of meansured torque as regression target"""
        return self._trajectory.measured_torque.T.flatten()

    def check_condition_number(self):
        cond_number = self._get_condition_number_with_traj(self._trajectory)
        return cond_number

    def _get_condition_number_with_traj(self, trajectory):
        regressor = self._get_regressor_from_trajectory(trajectory)
        parameter = self._get_parameter()
        regressor_base, _, _ = self._get_base_things_from_standard(
            regressor, parameter, 1e-6
        )
        cond_number = np.linalg.cond(regressor_base)
        return cond_number

    def get_regressor_and_parameter(self) -> Tuple[np.ndarray, Parameter]:
        """
        Computes the regressor matrix and retrieves the standard parameters.

        This method generates the regressor based on the stored trajectory and
        returns it alongside the full set of standard parameters, including
        those that may not be individually identifiable.

        Returns:
            tuple: A tuple containing:
                - regressor (np.ndarray): The computed regressor matrix.
                - parameter (Parameter): The standard parameter object
                  containing names and values.
        """
        regressor = self._get_regressor_from_trajectory(self._trajectory)
        parameter = self._get_parameter()
        return regressor, parameter

    def _get_regressor_from_trajectory(self, traj):
        q = np.array(traj.configuration)
        dq = np.array(traj.velocity)
        ddq = np.array(traj.acceleration)
        regressor = build_regressor_basic(self._robot, q, dq, ddq, self._config)
        return regressor

    def _get_parameter(self):
        names, values = get_standard_parameters(self._robot.model, self._config)
        return Parameter(names, values)

    def get_reduced_regressor_and_parameter(
        self, tol: float = 1e-6
    ) -> Tuple[np.ndarray, Parameter, Sequence[int]]:
        """
        Computes the reduced regressor and the corresponding parameters.

        This method performs column-pruning on the standard regressor matrix.
        Parameters corresponding to columns with a squared Euclidean norm below
        ``tol_e`` are considered unidentifiable (or unexcited) and are
        eliminated.

        Args:
            tol (float, optional): The tolerance threshold for the squared
                column norm. Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, Parameter]: A tuple containing:
                - regressor_reduced (np.ndarray): The regressor matrix with
                  low-excitation columns removed.
                - parameter_reduced (Parameter): The subset of parameters deemed
                  identifiable.
        """
        regressor_standard, parameter_standard = self.get_regressor_and_parameter()
        return self._reduce_regressor_and_parameter(
            regressor_standard, parameter_standard, tol
        )

    def _reduce_regressor_and_parameter(
        self, regressor_standard: np.ndarray, parameter_standard: Parameter, tolerance
    ):
        idx_eliminate, names_kept = get_index_eliminate(
            regressor_standard, parameter_standard.as_dict(), tol_e=tolerance
        )
        regressor_reduced = np.delete(regressor_standard, idx_eliminate, axis=1)
        values_kept = parameter_standard.get_values_by_names(names_kept)
        params_reduced = Parameter(names_kept, values_kept)
        return regressor_reduced, params_reduced, idx_eliminate

    def get_base_regressor_and_parameter(
        self, tol: float = 1e-6
    ) -> Tuple[np.ndarray, Parameter, np.ndarray]:
        """
        Computes the base regressor and the corresponding parameters.

        This method identifies linear dependencies in the regressor columns and
        groups dependent parameters. It returns a minimal set of base
        parameters, which are linear combinations of the standard parameters.

        Args:
            tol (float, optional): The tolerance threshold for the squared
                column norm. Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, Parameter, np.ndarray]: A tuple containing:
                - regressor_base (np.ndarray): The base regressor matrix (full
                  rank).
                - parameter_base (Parameter): An object containing the symbolic
                  names (linear combinations) of the base parameters.
                - The linear mapping from standard to base:
                      parameter_base = mapping * parameter_standard

        Note:
            The values in ``param_base`` may not be physically meaningful until
            estimated via least-squares, as they represent aggregate dynamic
            properties.
        """
        regressor_standard, parameter_standard = self.get_regressor_and_parameter()
        return self._get_base_things_from_standard(
            regressor_standard, parameter_standard, tol
        )

    def _get_base_things_from_standard(
        self, regressor_standard, parameter_standard, tolerance
    ):
        regressor_reduced, parameter_reduced, _ = self._reduce_regressor_and_parameter(
            regressor_standard, parameter_standard, tolerance
        )
        _, base_names_symbolic, base_idx = get_baseParams(
            regressor_reduced, list(parameter_reduced.names)
        )
        regressor_base = regressor_reduced[:, base_idx]

        mapping_standard2base = extract_linear_relation_between_base_and_standard(
            base_names_symbolic, parameter_standard.names
        )
        base_values = mapping_standard2base @ parameter_standard.get_values()
        parameter = Parameter(base_names_symbolic, base_values)
        return regressor_base, parameter, mapping_standard2base

    def random_like(self):
        return Regression(deepcopy(self._robot), self._trajectory.random_like())
