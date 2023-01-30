"""
Functions to load and analyse tokamak equilibria
"""

import numpy as np
from scipy import interpolate

from . import geqdsk
from .utils import critical, polygons

class Equilibrium:
    def __init__(self, R1D, Z1D, psi2D, wall = None):
        """
        
        Inputs
        ------

        R1D[nx]       1D array of major radius [m]
        Z1D[ny]       1D array of height [m]
        psi2D[nx,ny]  2D array of poloidal flux [Wb]

        Keywords
        --------

        wall = [(R0,Z0), (R1, Z1), ...]
               A list of coordinate pairs, defining the vessel wall.
               The wall is closed, so the last point connects to the first.

        """
        # Check the shape of the input
        assert len(R1D.shape) == 1
        assert len(Z1D.shape) == 1
        nx = R1D.shape[0]
        ny = Z1D.shape[0]
        assert len(psi2D.shape) == 2
        assert psi2D.shape == (nx, ny)
        
        # Convert 1D arrays of R, Z to 2D
        R, Z = np.meshgrid(R1D, Z1D, indexing="ij")

        self.R = R
        self.Z = Z
        self.psi = psi2D

        self.psi_func = interpolate.RectBivariateSpline(
            R1D, Z1D, psi2D
        )

        opoints, xpoints = critical.find_critical(
            R,
            Z,
            psi2D,
            1e-6, # atol
            1000  # maxits
        )

        self.opoints = opoints
        self.xpoints = xpoints

        if len(opoints) == 0:
            warnings.warn("No O-points found in Equilibrium input")
        else:
            self.psi_axis = opoints[0][2]  # Psi on O-point
            self.primary_opoint = (opoints[0][0], opoints[0][1])

        if len(xpoints) == 0:
            warnings.warn("No X-points found in Equilibrium input")
        else:
            self.psi_bdry = xpoints[0][2]  # Psi on primary X-point
            self.primary_xpoint = (xpoints[0][0], xpoints[0][1])

        # Bounding box for domain
        self.Rmin = min(R1D)
        self.Rmax = max(R1D)
        self.Zmin = min(Z1D)
        self.Zmax = max(Z1D)

        # Wall geometry. Note: should be anti-clockwise
        if wall is None:
            # No wall given, so add one which is just inside the domain edge
            offset = 1e-2  # in m
            wall = [
                (self.Rmin + offset, self.Zmin + offset),
                (self.Rmax - offset, self.Zmin + offset),
                (self.Rmax - offset, self.Zmax - offset),
                (self.Rmin + offset, self.Zmax - offset),
            ]
        elif len(wall) < 3:
            raise ValueError(
                f"Wall must be a polygon, so should have at least 3 points. Got "
                f"wall={wall}"
            )
        else:
            # Use wall location to eliminate X-points outside the wall
            if len(self.xpoints) > 1:
                filtered_xpoints = [self.xpoints[0]] # Keep primary
                for xpt in self.xpoints[1:]:
                    if not polygons.intersect(
                            [self.primary_opoint[0], xpt[0]], # R1
                            [self.primary_opoint[1], xpt[1]], # Z1
                            [p[0] for p in wall], # R2
                            [p[1] for p in wall], # Z2
                            closed1 = False,
                            closed2 = True):
                        # Line from primary O-point to this X-point does not intersect wall
                        # => Inside domain
                        filtered_xpoints.append(xpt)
                self.xpoints = filtered_xpoints

        if polygons.clockwise(wall):
            wall = wall[
                ::-1
            ]  # Reverse, without modifying input list (which .reverse() would)
        self.wall = wall
        self.wall_r = [p[0] for p in self.wall]
        self.wall_z = [p[1] for p in self.wall]

    def plot(self, axis=None, show=True):
        # Only import this if plotting needed
        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        levels = np.linspace(np.amin(self.psi), np.amax(self.psi), 100)

        axis.contour(self.R, self.Z, self.psi, levels=levels)
        axis.set_aspect("equal")
        axis.set_xlabel("Major radius [m]")
        axis.set_ylabel("Height [m]")

        for r, z, _ in self.xpoints:
            axis.plot(r, z, "rx")
        for r, z, _ in self.opoints:
            axis.plot(r, z, "go")
            
        if self.xpoints:
            axis.contour(self.R, self.Z, self.psi, levels=[self.psi_bdry], colors="r")

            # Add legend
            axis.plot([], [], "rx", label="X-points")
            axis.plot([], [], "r", label="Separatrix")
        if self.opoints:
            axis.plot([], [], "go", label="O-points")
            
        # Wall
        axis.plot(self.wall_r, self.wall_z, 'k')
        
        if show:
            plt.legend()
            plt.show()

        return axis

    def psiRZ(self, R, Z):
        """
        Return poloidal flux psi at given (R,Z) location
        """
        return self.psi_func(R, Z, grid=False)

    def limiter(self, start_rz, angle):
        """
        Find the location and poloidal flux at the limiter (wall)

        This is done by drawing a line from the start location outwards
        at the specified angle and finding the location of intersecton with the wall.
        
        start_rz = (R, Z)

        angle = 0 is outboard midplane
        angle = pi is inboard midplane

        Returns (R, Z, psi)
        """
        o_r = start_rz[0]
        o_z = start_rz[1]
        dist = 10. # m

        r, z = polygons.intersect(
            [o_r, o_r + dist * np.cos(angle)],
            [o_z, o_z + dist * np.sin(angle)],
            self.wall_r,  # r2
            self.wall_z,  # z2
            closed1 = False,
            closed2 = True
        )
        return (r, z, float(self.psiRZ(r, z)))

    def inner_limiter(self):
        """
        Inner midplane wall limiter psi
        Returns (R, Z, psi)
        """
        return self.limiter(self.primary_opoint, np.pi)

    def outer_limiter(self):
        """
        Outer midplane wall limiter psi
        Returns (R, Z, psi)
        """
        return self.limiter(self.primary_opoint, 0.0)

    def normalise_psi(self, psi):
        """
        Convert poloidal flux psi to normalised psi, using axis and separatrix psi.
        NOTE: This assumes that an X-point and an O-point were found.
        """
        return (psi - self.psi_axis) / (self.psi_bdry - self.psi_axis)

class EquilibriumType:
    def __init__(self, eq, psinorm_core = 0.95):

        # Normalised psi at core boundary
        self.psinorm_core = psinorm_core

        # Estimate outer limit on SOL psi
        in_r, in_z, in_psi = eq.inner_limiter()
        out_r, out_z, out_psi = eq.outer_limiter()

        # Convert to normalised psi
        in_psi_n = eq.normalise_psi(in_psi)
        out_psi_n = eq.normalise_psi(out_psi)

        self.psinorm_sol = min([in_psi_n, out_psi_n])

        # Find out how many X-points are in this range
        xpoints = [xpt for xpt in eq.xpoints if eq.normalise_psi(xpt[2]) < self.psinorm_sol]

        self.primary_opoint = (eq.primary_opoint[0], eq.primary_opoint[1])

        if len(xpoints) == 1:
            self.typestr = "snull"  # Single null
        elif len(xpoints) == 2:
            self.typestr = "dnull" # Double null
            self.psinorm_sol_inner = in_psi_n
            self.psinorm_sol = out_psi_n
        else:
            print("Number of X-points: {}".format(len(self.xpoints)))
            raise ValueError("Need to handle case with > 2 X-points")

        for xpt in xpoints:
            if xpt[1] > eq.primary_opoint[1]:
                # X-point is above O-point
                pf_r, pf_z, pf_psi = eq.limiter(xpt, np.pi/2)
                self.psinorm_pf_upper = eq.normalise_psi(pf_psi)
                self.xpoint_upper = (xpt[0], xpt[1])
            else:
                pf_r, pf_z, pf_psi = eq.limiter(xpt, -np.pi/2)
                self.psinorm_pf_lower = eq.normalise_psi(pf_psi)
                self.xpoint_lower = (xpt[0], xpt[1])

    def __str__(self):
        return "EquilibriumType(" + ",".join([attr + "=" + str(value) for attr, value in self.__dict__.items()]) + ")"

    def to_ingrid_dict(self):
        """
        Return a dict of settings for the INGRID grid generator
        """
        result = {}
        if self.typestr == "snull":
            # Single null

            lower = "xpoint_lower" in self.__dict__

            result["grid_settings"] = {"num_xpt": 1,
                                       # Psi levels
                                       "psi_1": self.psinorm_sol,
                                       "psi_core": self.psinorm_core,
                                       "psi_pf_1": self.psinorm_pf_lower if lower else self.psinorm_pf_upper,
                                       # magx coordinates
                                       "rmagx": self.primary_opoint[0],
                                       "zmagx": self.primary_opoint[1],
                                       # xpt coordinates
                                       "rxpt": self.xpoint_lower[0] if lower else self.xpoint_upper[0],
                                       "zxpt": self.xpoint_lower[1] if lower else self.xpoint_upper[1],
                                       "patch_generation": {
                                           "strike_pt_loc": 'limiter',
                                           "rmagx_shift": 0.0,
                                           "zmagx_shift": 0.0}}
        return result

    def to_hypnotoad_dict(self):
        """
        Return a dict of settings for the Hypnotoad grid generator
        """
        result = {}
        for attr in ["psinorm_core", "psinorm_sol", "psinorm_sol_inner", "psinorm_pf_lower", "psinorm_pf_upper"]:
            if attr in self.__dict__:
                result[attr] = getattr(self, attr)
        return result

def read_geqdsk(filename, cocos = 1):
    """
    Read a GEQDSK file, returning an Equilibrium object
    """
    
    with open(filename, 'r') as fh:
        data = geqdsk.read(fh)

    R1D = np.linspace(
        data["rleft"], data["rleft"] + data["rdim"], data["nx"], endpoint=True
    )

    Z1D = np.linspace(
        data["zmid"] - 0.5 * data["zdim"],
        data["zmid"] + 0.5 * data["zdim"],
        data["ny"],
        endpoint=True,
    )

    # Get the wall
    if "rlim" in data and "zlim" in data:
        wall = list(zip(data["rlim"], data["zlim"]))
    else:
        wall = None

    # Convert to Equilibrium
    return Equilibrium(R1D, Z1D, data["psi"], wall=wall)
