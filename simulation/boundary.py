import numpy as np

from simulation.state import State


class BoundaryCondition(object):
    def gradient(self, state: State, var: np.ndarray, grad: np.ndarray, axis: int) -> None:
        """Fill in the boundary of the gradient of a variable.

        The array in `grad` contains all interior elements of the gradient
        of `var` along the given axis.  This function injects all values on
        the boundary of that axis.
        """
        if axis not in (0, 1):
            raise Exception('Invalid axis provided')

        if axis == 1:
            self.gradient(state.replace(dx=state.dy, dy=state.dx), var.T, grad.T, 0)

        self._gradient(state, var, grad)

    def laplacian(self, state: State, var: np.ndarray, lapl: np.ndarray) -> None:
        """Fill in the boundary of laplacian of a variable."""
        self._gradient2(state, var, lapl)
        self._gradient2(state.replace(dx=state.dy, dy=state.dx), var.T, lapl.T)

    def _gradient(self, state: State, var: np.ndarray, grad: np.ndarray) -> None:
        raise NotImplementedError('_gradient not implemented.')

    def _gradient2(self, state: State, var: np.ndarray, lapl: np.ndarray) -> None:
        raise NotImplementedError('_laplacian not implemented.')


class Dirichlet(BoundaryCondition):
    def get_value(self, state: State) -> float:
        return state.config.getfloat('dirichlet', 'value', fallback=0)

    def _gradient(self, state: State, var: np.ndarray, grad: np.ndarray) -> None:
        h = state.dy
        value = self.get_value(state)

        # grad[0, :] += (var[0, :] - value) / h
        grad[-1, :] += (value - var[-1, :]) / h

    def _gradient2(self, state: State, var: np.ndarray, lapl: np.ndarray) -> None:
        h = state.dy
        value = self.get_value(state)

        lapl[0, :] += (value - 2 * var[0, :] + var[1, :]) / (h * h)
        lapl[-1, :] += (var[-2, :] - 2 * var[-1, :] + value) / (h * h)


class Neumann(BoundaryCondition):
    def _gradient(self, state: State, var: np.ndarray, grad: np.ndarray) -> None:
        # grad[0, :] = var[0, :]
        grad[-1, :] = var[-1, :]

    def _gradient2(self, state: State, var: np.ndarray, lapl: np.ndarray) -> None:
        h = state.dy

        lapl[0, :] = (-var[0, :] + var[1, :]) / (h * h)
        lapl[-1, :] = (var[-2, :] - var[-1, :]) / (h * h)
