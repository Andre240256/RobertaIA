import numpy as np
from typing import Callable

class edo_solver():
    r"""
    Implements a numerical solver for Ordinary Differential Equations (EDO) 
    using the 4th Order Runge-Kutta method (RK4).
    
    The solver approximates the solution to the initial value problem:
    \frac{dY}{dt} = F(t, Y, \tau_{motor}), \quad Y(t_0) = Y_0
    """

    def __init__(self, F: Callable[[float, np.ndarray, float], np.ndarray], h: float):
        r"""
        Initializes the solver with a target differential function and step size.

        :param F: A callable function F(t, Y, \tau) returning the derivative \frac{dY}{dt} 
                  as an np.ndarray of shape (2,).
        :param h: The integration step size (time increment \Delta t).
        """
        self.h = h
        self.F = F

    def passoRK4(self, ti: float, Yi: np.ndarray, tau_motor: float) -> np.ndarray:
        r"""
        Performs one integration step of the RK4 method to compute Y_{i+1}.

        Calculates the four increments:
        k_1 = F(t_i, Y_i, \tau)
        k_2 = F(t_i + \frac{h}{2}, Y_i + k_1 \frac{h}{2}, \tau)
        k_3 = F(t_i + \frac{h}{2}, Y_i + k_2 \frac{h}{2}, \tau)
        k_4 = F(t_i + h, Y_i + k_3 \cdot h, \tau)
        
        The result is approximated by:
        Y_{i+1} = Y_i + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)

        :param ti: Current time.
        :param Yi: State vector at time t_i of shape (2,).
        :param tau_motor: External control input (torque) applied to the system.
        :return: State vector Y_{i+1} at time t_i + h of shape (2,).
        """
        k1 = self.F(ti, Yi, tau_motor)
        k2 = self.F(ti + self.h/2, Yi + k1*self.h/2, tau_motor)
        k3 = self.F(ti + self.h/2, Yi + k2*self.h/2, tau_motor)
        k4 = self.F(ti + self.h, Yi + k3 * self.h, tau_motor)
        
        # Weighted average of slopes
        corr = (k1 + 2 * k2 + 2 * k3 + k4) * self.h / 6

        return Yi + corr