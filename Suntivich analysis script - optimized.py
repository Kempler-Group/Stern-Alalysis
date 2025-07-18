# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:19:47 2025

@author: stern
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit
import os
### fo
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "Arial" #MESS WITH THIS
plt.rcParams["mathtext.default"] = "regular"  # MAKES ITALICS NORMAL
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6


cl = "black" # color of dots

electrode_area = 0.13 #cm^2
R = 60  # resistance, 0 IF CORRECTED ON INSTRUMEN
comp = 0.015 # fractional %, iR compensation, if 85% on instrument, do remaining 15% here

Emin = 0.15
BLpoint = 0.39 ###baseline point
Emax = BLpoint + 0.01 #slightly highter than baseline to find the voltage closest to 0.36
Coverage = 0.08 # coverage you want to find the coverage at

### Au site density can be back calculated from "ideal" Cu coverage, change for respective ions
TCD = 440e-6 #C/cm^2, Theortical Charge Density (site occupancy) Cu on Au(111) = 440e-6, Ag on Au(111) = 222e-6
N = 2 # elementary charge (n = 2 for Cu2+)
mol_Au = TCD / (N * 96485.3321233100184) # C/cm^2, n, C/mol, will need to change 440e-6 (Cu site cov)
# Would be better to derive this using lattice parameters
#print(mol_Au, "mol cm^(-2)")

# File loading with dictionary and scan rates
library =       {  # Dictionary to store DataFrames and scan rates, the scan rates are the keys to the file path
    #25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-25_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt", 
    #50: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-50_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    #75: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-75_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    100: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-100_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    250: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-250_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    500: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-500_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    1000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-1000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    2000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-2000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    3000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-3000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    5000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-5000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    10000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-10000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    12500: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-12500_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    15000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-15000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    20000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-20000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    30000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-30000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    #50000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 3 mM Cu - 3w\CV-50000_Au(st008)_Pt_Cu[3mM]_100-3_mM_PCA-Cu_190225_02_CV_C01.mpt",
    # ... (rest of your paths)
    }

head=0
frames = {}  # Dictionary to store DataFrames, keyed by scan rate
for sr, file in library.items():
    try:
        df = pd.read_csv(file, delimiter='\t', header=head)
        frames[sr] = df  # Store the DataFrame with the scan rate as the key
    except FileNotFoundError:
        print(f"Error: File not found: {file}")
    except Exception as e:
        print(f"Error loading file {file}: {e}")

scan_rates = list(frames.keys()) # Extract scan rates from the dictionary keys
legend = [f"{sr/1000} V/s" for sr in scan_rates]  # Dynamic legend creation
cyclenum = [2] * len(frames)  #create a list of cycle numbers if needed
#plotColors = ["#000000", "#300c18", "#640d31", "#920346", "#b9095a", "#de086e", "#fd2084", "#ff65a0", "#ff8eb6", "#ffaeca", "#ffcbdc", "#ffe5ed", "#ffffff"] #for coverage vs potential plot
plotColors = ["#000000", "#1c0b0f", "#3e1923", "#5e2133", "#7e2642", "#9d2850", "#bb265d", "#d6276a", "#ec3177", "#f84d86", "#ff6996", "#ff86a9", "#ff9eba", "#ffb4c9", "#ffc8d7", "#ffdbe4", "#ffecf1", "#ffffff"]


def run_files(frames, plotColors, legend, scan_rates):  # Add scan_rates as a parameter
    '''
    function plots cov vs potential and returns the driving force and apparent rate constant for coverage =
    '''
    pca_E = []  #driving force for rate extraction
    P_ks = [] #list of electroadsorption constant
    
    
    plt.figure(figsize=(8,6))
    plt.subplots_adjust( top=0.955, bottom=0.16, left=0.155, right=0.975, hspace=0.2, wspace=0.2)

    plt.xlabel('$\mathit{E}$ (V vs $E_{rev}$)')
    plt.ylabel('$\mathit{\\theta}_\mathrm{Ag}$')
    plt.ylim(-0.005, 0.1)
    plt.tick_params(axis='both',which='both',direction='in')
    
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]  # Get the DataFrame using the scan rate as the key
        #print(frame)
        coverage, potential = Adsorption_Data(frame, [Emin, Emax], electrode_area, sr, BLpoint, cyclenum[i])  # Pass sr as scan rate
        print(coverage)
        plt.plot(potential, coverage, color=plotColors[i], label=legend[i])
        print(potential)
        pca_E.append(find_coverage(coverage, potential, Coverage))
        print(pca_E)
        P_ks.append(lookup_k(frame, pca_E[i], electrode_area, Coverage, BLpoint, cyclenum[i]))
        
   
    plt.legend()
    
    print(pca_E)                 
    print(P_ks)
    
    get_K_app(pca_E, P_ks, cl)
    #get_K_app2(pca_E, P_ks)



def get_K_app(pca_E, P_ks, cl):  # Pass Coverage and cl as arguments
    """Calculates and plots the apparent rate constant.

    Args:
        pca_E (list/np.ndarray): Driving forces.
        P_ks (list/np.ndarray): Rate constants.
        Coverage (float): Coverage value.
        cl (str): Color for the data points.
    """

    P_ksi = np.array(pca_E) - pca_E[0]  # Use NumPy for driving force calculation
    C = Coverage
    a = (C * (1 - C))**0.5

    xData = P_ksi # driving force (greek letter ksi)
    yData = P_ks #apparant rate

    x1 = np.linspace(-0.1, 0, 10000)

    # Constants (define these outside the function if they are truly constant)
    e = 1.60217663e-19  # C
    kb = 1.380649e-23  # J K-1
    T = 298.15  # K
    con = e / (2 * kb * T)  # V

    def func(x, k):
        return (k * a) * (np.exp(-(con * x)) - np.exp((con * x)))#eqn from suntivisch paper
    #curve fits takes 3 things and gives (look up) apparent rate const and error 
    popt, pcov = curve_fit(func, xData, yData, p0=[1])  # Provide an initial guess
    k_app = popt[0]  # Extract the k_app value
    err = pcov  # Keep the error matrix
    print('the apparent rate constant is: [', k_app, '] and the error is ', err) # Print k_app and error matrix
    print(C)

    plt.figure(figsize=(6, 7))
    plt.tick_params(axis='both', which='both', direction='in')
    plt.subplots_adjust(top=0.965, bottom=0.135, left=0.235, right=0.98, hspace=0.2, wspace=0.2)

    plt.plot(P_ksi, P_ks, 'o', color=cl, markersize=10)
    plt.plot(x1, func(x1, k_app), color='black')  # Use k_app directly
    plt.tick_params(axis='both', which='both', direction='in')
    plt.yscale('log')
    plt.ylim([5e-3, 5e+2])
    plt.xlim([-0.10, 0.01])

    plt.xlabel('$\mathit{\\xi}$ (V vs $E_{eq}$)')
    plt.ylabel('|$\mathit{k}_\mathrm{app}$| ($\mathregular{s^{-1}}$)')
    # Prepare the equations to display
    s = "$\mathregular{s^{-1}}$"
    K_o = '$\mathit{k}_\mathrm{0}$'
    K_app_str = f"{K_o} = {k_app:.2g} {s}"  # Formatted string for k_app
    theta = "$\mathit{\\theta}_\mathrm{Ag}$"
    coverage_str = f" {theta} = {Coverage}"

    plt.text(0.05, 0.10, K_app_str, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.text(0.05, 0.15, coverage_str, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.show()
    
    return k_app, err # Return k_app and err



def lookup_k(frame, potential, area, coverage, baseLine, cycle):
    '''This function takes a frame and slices to the adsorption trace
    Then it finds the first value greater than the specified potential and converts
    the current measurement at this point to an apparent rate constant, for an assumed
    value of n (# of eletrons transfered)
    
    k is in units of s-1 and is normalized to the number of surface sites
    i = nFA * [k_ox(theta) - k_red(1-theta)] * theta*
    
    units only make sense if theta is in mol/cm^2
    '''
    
    """
    Calculates the apparent rate constant (k) at a given potential.

    Args:
        frame (pd.DataFrame): The DataFrame containing the CV data.
        potential (float): The target potential.
        area (float): Electrode area.
        coverage (float): Target coverage.
        baseLine (float): Baseline potential.
        cycle (int): Cycle number.
        N (int, optional): Number of electrons transferred. Defaults to 2.
        R (float, optional): Uncompensated resistance. Defaults to 0.
        mol_Au (float): Surface site density of gold.

    Returns:
        float: The apparent rate constant k, or None if the potential is not found.
    """

    if mol_Au is None:
        raise ValueError("mol_Au must be provided.")

    frame = frame[frame['cycle number'] == cycle]
    frame = frame.iloc[:round(len(frame) / 2)].copy()

    frame['<J>/A cm^-2'] = frame['<I>/mA'] / (area * 1000)
    frame['E_corr/V'] = frame['Ewe/V'] - (frame['<I>/mA'] / 1000) * R * comp

    E = frame['E_corr/V'].values  # NumPy array for potential
    i0 = np.argmin(np.abs(E - baseLine))
    i0min = max(0, i0 - 2)
    i0max = min(len(frame) - 1, i0 + 2)
    bL = frame.iloc[i0min:i0max]['<J>/A cm^-2'].mean()
    corrFrame_J = frame['<J>/A cm^-2'].values - bL  # NumPy array for corrected current
    print(type(potential))
    print(type(E))
    # Find the *first* index where E < potential
    n_candidates = np.where(E < potential)[0]

    if n_candidates.size > 0:
        n = n_candidates[0]  # First index
        j = corrFrame_J[n]
        print('Current = ', j)
        k = abs(j) / (N * 96485 * (1 - coverage) * mol_Au)
        print('k_app = ', k)
        return k
    else:
        return None

def find_coverage(thetas, Es, coverage): #Q-list is thetas, E -> Es, and coverage is taarget cov (comparing accross scan rates)
   #Q_list is coverage, Es = corrected potential, target coverage (arb) 
    '''This function finds the potential for which a given target coverage is reached'''
    
    for n, theta in enumerate(thetas):
        if max(thetas) < coverage:
            raise ValueError(f"Target coverage '{coverage}' does not exist within the given coverage values, chose a LOWER coverage or remove the scans that do not reach the desired coverage (may affect rate).")   
        elif theta > coverage:
            #print(Es.iloc[n]) ### turned off bec a million messages pop up
            return Es.iloc[n]
        else:
            continue


def Adsorption_Data(frame, bounds, electrode_area, scanRate, baseLine, cycle):  # Added R as a parameter with default
    """
    Corrects data to a baseline and calculates adsorption coverage.

    Args:
        frame (pd.DataFrame): The DataFrame containing the CV data.
        bounds (list): The potential bounds for the adsorption region.
        electrode_area (float): The electrode area in cm^2.
        scanRate (float): The scan rate in V/s.
        baseLine (float): The baseline potential.
        cycle (int): The cycle number to analyze.
        TCD (float, optional): The normalization charge density. Defaults to 440e-6.
        R (float, optional): The uncompensated resistance. Defaults to 0.

    Returns:
        tuple: A tuple containing the list of adsorption coverages (Q_list) and the corrected potentials (E).
    """

    frame = frame[frame['cycle number'] == 2]
    frame = frame.iloc[:round(len(frame) / 2)].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Calculate current density (vectorized)
    frame['<J>/A cm^-2'] = frame['<I>/mA'] / (electrode_area * 1000)

    # Correct potential for IR drop (vectorized)
    frame['E_corr/V'] = frame['Ewe/V'] - (frame['<I>/mA'] / 1000) * R * comp

    # Trim frame (vectorized)
    frame = frame[(frame['E_corr/V'] < max(bounds)) & (frame['E_corr/V'] > min(bounds))]
    frame.reset_index(drop=True, inplace=True)

    E = frame['E_corr/V']

    # Baseline correction (more efficient)
    i0 = np.argmin(np.abs(E - baseLine))
    i0min = max(0, i0 - 2)
    i0max = min(len(frame) - 1, i0 + 2)
    baseLineArray = frame.iloc[i0min:i0max]['<J>/A cm^-2']
    bL = baseLineArray.mean()
    corrFrame_J = frame['<J>/A cm^-2'] - bL


    # Calculate adsorption coverage (Corrected and more efficient)
    Q_list = []
    Q = 0
    time = frame['time/s'].values
    corrFrame_J_values = corrFrame_J.values
    #simpler/different intergration that doesnt use np.trapz
    for i in range(1, len(frame)):
        dt = time[i] - time[i - 1] #time differeenctiated over
        #intergration (make box with hight J + J+1, and length dt, then divide to get trangle (trapaziod integration bec in a for-loop)
        Q += (corrFrame_J_values[i] + corrFrame_J_values[i - 1]) * dt / 2 
        adsorption = -Q / TCD #normalize to coverage
        Q_list.append(adsorption)

    # E has one more element than Q_list, so we slice E to match
    E_matched = E.iloc[1:] # slice E to match the length of Q_list

    return Q_list, E_matched  # Return E_matched



run_files(frames, plotColors, legend, scan_rates)