'''
GOAL:   To familiarize myself with the workings of ASE, generating a simulation, and running a simulation.
HOW:    Take the code from PySAGES water example and refit it to work for NaCl unit cells.
'''

import numpy as np
from ase import Atoms, units
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.md import Langevin


def generate_simulation(tag="tip3p", write_output=True):

    # angleHOH and rOH are imported variables, so names must remain as the is

    x = angleHOH * np.pi / 180 / 2
    pos = [
        [0, 0, 0],  # rOH is the distance between oxygen and hydrogen atoms in water
        [0, rOH * np.cos(x), rOH * np.sin(x)],
        # [0, rOH * np.cos(x), -rOH * np.sin(x)],
    ]
    atoms = Atoms("Na+Cl", positions=pos)


    # atoms = Atoms("OH2", positions=pos)

    vol = ((18.01528 / 6.022140857e23) / (0.9982 / 1e24)) ** (1 / 3.0)
    atoms.set_cell((vol, vol, vol))
    atoms.center()

    atoms = atoms.repeat((3, 3, 3))
    atoms.set_pbc(True)

    atoms.constraints = FixBondLengths(
        # [(3 * i + j, 3 * i + (j + 1) % 3) for i in range(3**3) for j in [0, 1, 2]]
        [(2 * i + j, 2 * i + (j + 1) % 2) for i in range(3**3) for j in [0, 1]]
    )

    T = 300 * units.kB
    atoms.calc = TIP3P(rc=4.5)
    logfile = tag + ".log" if write_output else None
    md = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.01, logfile=logfile)

    if write_output:
        traj = Trajectory(tag + ".traj", "w", atoms)
        md.attach(traj.write, interval=1)

    return md



generate_simulation()