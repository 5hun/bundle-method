from typing import Tuple, Union

import numpy as np
import cvxpy as cp

class ProximalBundleMethod:
    r"""
    Proximal bundle method
    
    The impolementations is based on the figure 7.3 of [1]_ .
    
    Parameters
    ----------
    n: int
        number of parameters
    alpha: float
        proximal parameter
    gamma: float
    sense: function
        max or min

    Attributes
    ----------
    x: cvxpy.Variable
        read only
    custom_constraints: list of cvxpy.constraints.constraint.Constraint
        constraints define the feasible region of x.

    References
    ----------
    .. [1] Andrzej Ruszczynski, "Nonlinear Optimization", Princeton University Press, 2006.

    """

    _SENSE2DEFAULT = {
        max: float("inf"),
        min: float("-inf")
    }
    
    def __init__(
            self, n: int, alpha: float = 1, gamma: float = 0.5, sense=max
    ):
        self._alpha = alpha
        self._gamma = gamma
        self._sense = sense

        self._x = cp.Variable(n)
        self._y = cp.Variable()
        self.custom_constraints = []
        self._cuts = []
        self._consts = []

        self._center = None
        self._center_obj = None
    
        self._v = self._SENSE2DEFAULT[self._sense]

    @property
    def x(self):
        return self._x

    def _evaluate(self, x) -> float:
        if self._sense == max:
            op = min
        else:
            op = max
        return op(
            (obj + g @ (x - x0) for obj, x0, g in self._cuts),
            default=self._SENSE2DEFAULT[self._sense]
        )

    def _get_cut_const(self, obj, x, grad):
        if self._sense == max:
            return self._y <= obj - grad @ x + grad @ self._x
        else:
            return self._y >= obj - grad @ x + grad @ self._x
    
    def step(self, obj, x, subgrad) -> np.ndarray:
        r"""
        Add cut and get new point.
        
        Parameters
        ----------
        obj: float
            objective value
        x: numpy.ndarray
        subgrad: numpy.ndarray
            subgradient at x

        Returns
        -------
        numpy.ndarray
            the next point to evaluate

        """

        self._add_cut(obj, x, subgrad)
        return self._solve()

    def _add_cut(self, obj, x, subgrad) -> None:
        if self._center is None:
            self._consts.append(self._get_cut_const(obj, x, subgrad))
            self._cuts.append((obj, x, subgrad))
            self._center = x
            self._center_obj = obj
        else:
            tgt = (1 - self._gamma) * self._center_obj + self._gamma * self._evaluate(x)
            if (self._sense == max and obj >= tgt) or (self._sense == min and obj <= tgt):
                # Serious Step
                self._center = x
                self._center_obj = obj
            if (self._sense == max and obj < self._v) or (self._sense == min and obj > self._v):
                self._consts.append(self._get_cut_const(obj, x, subgrad))
                self._cuts.append((obj, x, subgrad))

    def _solve(self) -> Tuple[np.ndarray, bool]:
        if self._sense == max:
            obj = cp.Maximize(self._y - 0.5 * self._alpha * cp.sum_squares(self._x - self._center))
        else:
            obj = cp.Minimize(self._y + 0.5 * self._alpha * cp.sum_squares(self._x - self._center))
        prob = cp.Problem(obj, self.custom_constraints + self._consts)
        prob.solve()
        self._v = self._y.value
        idx = [i for i, c in enumerate(self._consts) if c.dual_value > 1e-8]
        self._consts = [self._consts[i] for i in idx]
        self._cuts = [self._cuts[i] for i in idx]
        return self._x.value
