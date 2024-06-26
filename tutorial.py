import numpy as np
from ase import Atoms, units
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.md import Langevin
from ase.calculators.vasp import Vasp

'''
GOAL:   To familiarize myself with the workings of ASE, generating a simulation, and running a simulation.
HOW:    Take the code from PySAGES water example and refit it to work for NaCl unit cells.
'''


rNaCl = 2.8    # distasnce between Na and Cl atoms in Angstroms
angleNaCl = 180

def generate_simulation(tag="custom_NaCl", write_output=True):
    pos = [
        [0, 0, 0],
        [0, 0, rNaCl]
    ]

    atoms = Atoms("NaCl", positions=pos)

    vol = 20
    atoms.set_cell((vol, vol, vol))
    atoms.center()

    atoms.set_pbc(True)

    T = 300 * units.kB
    calc = Vasp(
        xc='PBE',
        encut=300,
        kpts=(1,1,1),
        ibrion=2,
        nsw=50,
        potim=1.0,
        ismear=0,
        sigma=0.05,
        lreal='Auto'
    )
    # atoms.calc = calc.initialize(atoms)
    calc.initialize(atoms)
    atoms.set_calculator(calc)
    calc.write_input(atoms)

    logfile = tag + ".log" if write_output else None
    logfile_obj = open(logfile, 'a')
    logfile_obj.truncate(0)
    logfile_obj.close()
    dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.01, logfile=logfile)


    if write_output:
        traj = Trajectory(tag + ".traj", "w", atoms)
        dyn.attach(traj.write, interval=1)
    
    dyn.run(5)

    return dyn



generate_simulation()