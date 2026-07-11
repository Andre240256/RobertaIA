class edo_solver():

    def __init__(self, F, h):
        self.h = h
        self.F = F


    def passoRK4(self, ti, Yi, tau_motor):
        k1 = self.F(ti, Yi, tau_motor)
        k2 = self.F(ti + self.h/2, Yi + k1*self.h/2, tau_motor)
        k3 = self.F(ti + self.h/2, Yi + k2*self.h/2, tau_motor)
        k4 = self.F(ti + self.h, Yi + k3 * self.h, tau_motor)
        corr = (k1 + 2 * k2 + 2 * k3 + k4) * self.h/6

        return Yi + corr
    
