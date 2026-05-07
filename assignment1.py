import pandas as pd
import subprocess  # subprocess is usefull for running shell commands from python
import sys  # sys is usefull for system specific parameters and functions
import numpy as np  # numpy is usefull for vectorized operations
import scipy.constants as sc  # scipy.constants has many useful constants such as avogadro's number or boltzmann's constant
import matplotlib.pyplot as plt  # plotting library
from scipy import stats  # statistics has many useful functions for statistical analysis

### FUNCTIONS FROM JUPITER NOTEBOOK ### 
def distance(pos: np.ndarray) -> np.ndarray:
    # This is the distance4 function from above
    # Initialize a 2D array to store squared pairwise distances
    r2 = np.zeros((pos.shape[0], pos.shape[0]))
    for i, pos_i in enumerate(pos):
        # loop over particle i
        # Calculate the squared differences between point i and all previous points
        # but exclude all j larger than i.
        r2[i, :i] = np.sum((pos_i - pos[:i, :]) ** 2, axis=1)
    # retrieve all j larger than i by transposing the matrix
    # and adding it to the original matrix
    r2 += r2.T
    r = np.sqrt(r2)
    return r

def periodic_BC(pos: np.ndarray, L: float) -> np.ndarray:    
    # This function was written with the help of Claude and my memory from last year
    # I brainstormed with claude to make the PBC process faster 
    pos_shifted = ( pos + L/2) % L - L/2 # shift positions to be between -L/2 and L/2

    return distance(pos_shifted)

def totalEnergy(pos: np.ndarray) -> float:
    ### Define parameters and cutoff potential energy 
    sigma = 3.78e-10 #[m] length parameter for CH4 
    kb = 1.38e-23 #[J/K] Boltzmann constant 
    epsilon = 179 * kb # [J] Lennard-Jones epsilon parameter 
    rho = N / (L * 1e-10)**3 # [particles/m^3] number density of the system
    r_c = 2.5 * sigma # [m] cut-off distance 
    L_ang = 30e-10 # [m] length of box wall in meters
    Av = 6.022e23 # Avogadros constant 
    
    # trim distances and convert to meters
    r = periodic_BC(pos, L)
    r = np.tril(r)
    r = r * 1e-10 # convert from angstrom to meters
    r = np.where(r == 0, np.inf, r)
   
    # split above and below cut-off distances
    r_below_cutoff = np.where(r < r_c, r, np.inf) # set distances above cut-off to infinity so they do not contribute to the energy

    # build the formula fo reverything below the cut-off distance
    sr_below = sigma / r_below_cutoff
    sr_below2 = sr_below * sr_below
    sr_below6 = sr_below2 * sr_below2 * sr_below2
    sr_below12 = sr_below6 * sr_below6
    u_below_matrix = 4 * epsilon * (sr_below12 - sr_below6)
 
    # tail correction 
    u_tail = ( 8 / 3) * np.pi * rho * epsilon * sigma**3 * (1/3 * (sigma / r_c) ** 9 - (sigma / r_c) ** 3)
   
    # you might require additional inputs to this functions
    energy = np.sum(u_below_matrix) + u_tail
    energy_JpMol = energy / (N / Av) # convert energy from Joules to Joules per mole
    return energy, energy_JpMol

def periodic_BC_single(pos: np.ndarray, L: float, i: int) -> np.ndarray:
    # this function is similar to the total BC but only calculates the distances for 1 particle to make code faster. 
    delta = pos[i, :] - pos # get positions relative to particle i (matrix is N x 3 )
    delta -= L * np.round(delta / L) # apply the MIC
    r = np.sqrt(np.sum(delta**2, axis=1)) # Calculates relative distance between particle i and all other particles (matrix is N x 1)
    r[i] = np.inf # set distance from i to i to infinity. 
    return r

def singleParticleEnergy(pos: np.ndarray, i: int) -> float:
    ### Define parameters and cutoff potential energy 
    N = len(pos) # number of particles in the system
    L = 30
    sigma = 3.78e-10 #[m] length parameter for CH4 
    kb = 1.38e-23 #[J/K] Boltzmann constant 
    epsilon = 179 * kb # [J] Lennard-Jones epsilon parameter 
    rho = N / (L * 1e-10)**3 # [particles/m^3] number density of the system
    r_c = 2.5 * sigma # [m] cut-off distance 
    L_ang = 30e-10 # [m] length of box wall in meters
    Av = 6.022e23 # Avogadros constant 

    # trim distances and convert to meters
    r = periodic_BC_single(pos, L, i) # Apply MIC to the particle of interest
    r = r * 1e-10 # convert from angstrom to meters
    r = np.where(r == 0, np.inf, r)

    # split above and below cut-off distances
    r_below_cutoff = np.where(r < r_c, r, np.inf) # set distances above cut-off to infinity so they do not contribute to the energy

    # build the formula fo reverything below the cut-off distance
    sr_below = sigma / r_below_cutoff
    sr_below2 = sr_below * sr_below
    sr_below6 = sr_below2 * sr_below2 * sr_below2
    sr_below12 = sr_below6 * sr_below6
    u_below_matrix = 4 * epsilon * (sr_below12 - sr_below6)
 
    # tail correction 
    u_tail = ( 8 / 3) * np.pi * rho * epsilon * sigma**3 * (1/3 * (sigma / r_c) ** 9 - (sigma / r_c) ** 3)
   
    # you might require additional inputs to this functions
    energy = np.sum(u_below_matrix) + u_tail
    energy_JpMol = energy / (N / Av) # convert energy from Joules to Joules per mole

    return energy, energy_JpMol

def observables(pos: np.ndarray) -> tuple[float, float]:
    ### Define parameters and cutoff potential energy 
    sigma = 3.78e-10 #[m] length parameter for CH4 
    N = len(pos) # number of particles in the system
    L = 30
    kb = 1.38e-23 #[J/K] Boltzmann constant 
    epsilon = 179 * kb # [J] Lennard-Jones epsilon parameter 
    rho = N / (L * 1e-10)**3 # [particles/m^3] number density of the system
    r_c = 2.5 * sigma # [m] cut-off distance 
    Av = 6.022e23 # Avogadros constant 
    T = 150 # [K] temperature in kelvin 
    V = (L * 1e-10)**3 # [m^3] volume 

    # trim distances and convert to meters
    r = periodic_BC(pos, L)
    r = np.tril(r)
    r = r * 1e-10 # convert from angstrom to meters
    r = np.where(r == 0, np.inf, r)
   
    # split above and below cut-off distances
    r_below_cutoff = np.where(r < r_c, r, np.inf) # set distances above cut-off to infinity so they do not contribute to the energy

    # build the formula fo reverything below the cut-off distance
    sr_below = sigma / r_below_cutoff
    sr_below2 = sr_below * sr_below
    sr_below6 = sr_below2 * sr_below2 * sr_below2
    sr_below12 = sr_below6 * sr_below6
    u_below_matrix = 4 * epsilon * (sr_below12 - sr_below6)
 
    # tail correction 
    u_tail = ( 8 / 3) * np.pi * rho * epsilon * sigma**3 * (1/3 * (sigma / r_c) ** 9 - (sigma / r_c) ** 3)
   
    # Function to calculate the derivative of the Lennard-Jones potential with respect to r
    def DUDr(r: float) -> float:
        r2 = r * r 
        r6 = r2 * r2 * r2
        r12 = r6 * r6
        r13 = r12 * r
        r7 = r6 * r
        du_dr = 4 * epsilon * ( (-12) * ( sigma ** 12 / r13) + 6 * ( sigma ** 6 / r7) )
        return du_dr
    
    
    # Calculate the virial pressure 
    product = r_below_cutoff * DUDr(r_below_cutoff)
    sum = np.nansum(product)     # Claude was used here to help debug the summation and verify whether calculated pressure seems reasonable. 
    ideal = rho * kb * T
    virial = (1 / (3 * V)) * sum
    P = ideal - virial 

    # total energy and pressure in SI units:
    energy = np.sum(u_below_matrix) + u_tail
    pressure = P
    
    # in convenient units: 
    #energy_kJpMol = ( energy / (N / Av) ) / 1000 # convert energy from J to kJ/mol
    #pressure_bar = pressure / 1e5 # convert pressure from Pa to bar

    return energy, pressure

def translate(pos: np.ndarray) -> np.ndarray:
    i = np.random.randint(0, len(pos)) # select  random particle index
    selected = pos[i] # get the position of the selected particle in meters 

    # get a random displacement vector and move the particle 
    move_max = 0.5  # [m] maximum move distance 
    displacement = (np.random.uniform(-move_max, move_max, size=3)) # get randomdisplacement vector between min and max value 
    selected_new = selected + displacement # position of the displaced particle 

    #displace the particle in the box. 
    pos_new = np.copy(pos) # create a copy of the original positions to modify
    pos_new[i] = selected_new # update the position of the

    # calculate single particle energy for old and new position (PBC included in the energy function)
    energy_old, energy_old_pm = singleParticleEnergy(pos, i)
    energy_new, energy_new_pm = singleParticleEnergy(pos_new, i)
    delta_U = energy_new - energy_old

    # determine whether to accept the move using the Metropolis criterion
    T = 150 # [K] temperature in kelvin
    kb = 1.38e-23 #[J/K] Boltzmann constant
    beta = 1 / (kb * T) # inverse temperature
    treshold = np.exp(-beta * delta_U) # Metropolis acceptance criterion treshold
    rand = np.random.uniform(0, 1) # get a random number between 0 and 1

    if delta_U < 0: 
        accept = True
    elif rand < treshold:
        accept = True
    else:
        accept = False
       # print("Move rejected. Energy change:", delta_U, "Joules, random number was:", rand, "treshold was:", treshold)

    if accept:
        pos = pos_new # update the positions to the new positions if the move is accepted
        #print("Move accepted. Energy change:", delta_U, "Joules, random number was:", rand, "treshold was:", treshold)

    return pos, accept 

def startConf(nParticles: int, density: float, nEquilibrate: int) -> tuple[np.ndarray, float]:
    # you might require additional inputs to this functions
    # this function is a perfect place to use scipy.constants.N_A ect.
    molar_mass = 16.04 / 1000 # [kg/mol] molar mass of methane
    n_av = sc.N_A # Avogadro's number in particles per mole
    # rho = mass / volume --> volume = mass / rho --> volume = (N * molar_mass ) / rho 
    V = ( ( nParticles / n_av ) * molar_mass) / density # [m^3] 
    lbox = V ** (1/3) # [m] 
    L_ang = lbox * 1e10 # [angstrom] box length in angstrom 
    print("box length in angstrom:", lbox * 1e10)

    # generate initial confifuration 
    pos = np.random.uniform(0, L_ang, size=(nParticles, 3)) # Generate random XYZ coordinates for nParticles withing the box

    accepted_moves = 0
    for i in range(nEquilibrate):
        pos, succes = translate(pos)
        if succes: 
            accepted_moves += 1
    acceptance_rate = accepted_moves / nEquilibrate

    return pos, lbox, acceptance_rate

def averages(observable: np.ndarray) -> tuple[float, float]:
    # you might require additional inputs to this functions
    mean = np.mean(observable)
    std = np.std(observable)

    ### t-value at 95% and uncertainty: 
    n = len(observable)
    t_value = stats.t.ppf(0.975, df=n-1)
    uncertainty = t_value * ( std / np.sqrt(n))
    
    return mean, uncertainty

def MC_NVT(nParticles: int, density: float, nInitCycle: int,
           nCycle: int, nSpacing: int) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    # you might require additional inputs to this functions
    
    # get initial configuration
    pos, lbox, acc_rate_init = startConf(nParticles, density, nInitCycle)
    
    pressure = np.zeros(nCycle)
    energy = np.zeros_like(pressure)
    tries = 0
    successes = 0

    for i in range(nCycle):
        for _ in range(nSpacing):
            # translate the system nSpacing times.
            # We use _ to indicate that this variable is not used in any way.
            # This is a common python convention to indicate that the variable is not used.
            pos, success = translate(pos)
            tries += 1
            if success:
                successes += 1
        # compute the observables
        energy[i], pressure[i] = observables(pos)
    
    # compute averages
    pm, pu = averages(pressure)
    em, eu = averages(energy)
    
    acc_rate = successes / tries # compute acceptance rate

    return pm, pu, em, eu, acc_rate, pressure, energy

def main():
    # Expected 5 arguments + 1 for script name
    if len(sys.argv) != 6:
        print("Usage: python code.py <nParticles> <density> <nInitCycle> <nCycle> <nSpacing>")
        sys.exit(1)

    errors = []

    # Individual parsing with detailed error tracking
    try:
        nParticles = int(sys.argv[1])
    except ValueError:
        errors.append(f"Invalid nParticles: expected an integer, got '{sys.argv[1]}'.")

    try:
        density = float(sys.argv[2])
    except ValueError:
        errors.append(f"Invalid density: expected a float, got '{sys.argv[2]}'.")

    try:
        nInitCycle = int(sys.argv[3])
    except ValueError:
        errors.append(f"Invalid nInitCycle: expected an integer, got '{sys.argv[3]}'.")

    try:
        nCycle = int(sys.argv[4])
    except ValueError:
        errors.append(f"Invalid nCycle: expected an integer, got '{sys.argv[4]}'.")

    try:
        nSpacing = int(sys.argv[5])
    except ValueError:
        errors.append(f"Invalid nSpacing: expected an integer, got '{sys.argv[5]}'.")

    # If there were any parsing errors, print and exit
    if errors:
        print("\nErrors detected:")
        for error in errors:
            print(" -", error)
        sys.exit(1)

    # Output the collected variables
    print("\nCollected Inputs:")
    print(f"nParticles: {nParticles}")
    print(f"density: {density}")
    print(f"nInitCycle: {nInitCycle}")
    print(f"nCycle: {nCycle}")
    print(f"nSpacing: {nSpacing}")

    pm, pu, em, eu, acc_rate, pressure, energy = MC_NVT(nParticles, density, nInitCycle, nCycle, nSpacing)

    # create a pandas dataframe
    # save the dataframe as a file
    results_df = pd.DataFrame({
    'mean_pressure': [pm],
    'pressure_uncertainty': [pu], 
    'mean_energy': [em],
    'energy_uncertainty': [eu],
    'acceptance_rate': [acc_rate],
    'nParticles': [nParticles],
    'density': [density],
    'nInitCycle': [nInitCycle],
    'nCycle': [nCycle],
    'nSpacing': [nSpacing]
    })

    # Save the dataframe as a file
    filename = f"MC_results_N{nParticles}_rho{density}_cycles{nCycle}.csv"
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()