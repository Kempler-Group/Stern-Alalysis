# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:14:58 2025

@author: Owner
"""

""" is way more dependnant on the bounds of the piece being fit than the approximation """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


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
lw = 4 #line width


conc = 3 # mM conc
concentration = conc/1000 # Molar 10 mM = 0.010 M

SA = 0.137 #cm^2
R = 50.859 # resistance, 0 IF CORRECTED ON INSTRUMEN
comp = 0.15 # fractional %, iR compensation, if 85% on instrument, do remaining 15% here

BLpoint = 0.44 ###baseline point
coverage = 0.05 # 5%, the low coverages for which a linerar approx would work and where we want to compare isotherms at

### Au site density can be back calculated from "ideal" Cu coverage, change for respective ions
TCD = 440e-6 #C/cm^2, Theortical Charge Density (site occupancy) Cu on Au(111) = 440e-6, Ag on Au(111) = 220e-6
N = 2 # elementary charge (n = 2 for Cu2+)
mol_Au = TCD / (N * 96485) # C/cm^2, n, C/mol, will need to change 440e-6 (Cu site cov)
# Would be better to derive this using lattice parameters
#print(mol_Au, "mol cm^(-2)")

colors = ["black", "blue", "dodgerblue", "skyblue"] # color of dots

# File loading with dictionary and scan rates
library = {  # Dictionary to store DataFrames and scan rates, the scan rates are the keys to the file path
    50: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\19-02-25 10 mM Cu - 3w\CV-50_Au(st008)_Pt_Cu[10mM]_100-10_mM_PCA-Cu_190225_02_CV_C01.mpt",
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

SR = list(frames.keys()) # Extract scan rates from the dictionary keys
legend = [f"{sr/1000} V/s" for sr in SR]  # Dynamic legend creation
cyclenum = [2] * len(frames)  #create a list of cycle numbers if needed


RT = 298.15 * 8.314 #J/mol 
F = 96485 #C/mol
p = RT/F 

def frumkin(params, x): #solved for potential
    g,  e = params
    
    return ((-p) *  (np.log( x / ( 1 - x )) + g*(x-0.5) -np.log(concentration)/N ) + e)
    # Frumkin: Θ = (K_eq*C_A)/(exp(g*Θ) - K_eq*C_A), Langmuir: Θ = (K_eq*C_A)/(1 - K_eq*C_A) solve for E

def plot_fits(frames, rates, cycle, lednames, colors, electrode_area):  #rates = [10, 20, 30, 40, 60, 80, 100], rates from scan in frames
    
    coverage, potential, fitted_params, Emin, Emax = get_frumkin()
    g, E_form = fitted_params
    coverage = coverage 

    #plot the frumkin fit
    
    g_text = "$\mathit{g}$"
    interaction_param = f"{g_text} = {g:.1f} "
    gamma =  (g*RT)/(2*1000)
    symbol = "$\mathit{γ}$"
    ef = "$\mathit{E}_{f}$"
    E_rev = "$\mathit{E}_{rev}$"
    formal_pot = f"{ef} = {E_form:.4f} V vs {E_rev}"
    interaction_energy = f"{symbol} = {gamma:.1f} kJ/mol"
    
    # plt.plot(eq) in nice box
    plt.figure(figsize=(8,6.5))
    plt.subplots_adjust( top=0.930, bottom=0.155, left=0.150, right=0.965, hspace=0.2, wspace=0.2)
    # Display the equations on the plot
    plt.text(0.45, 0.97, interaction_param, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.text(0.45, 0.90, interaction_energy, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.text(0.45, 0.83, formal_pot ,transform = plt.gca().transAxes, fontsize=24, verticalalignment = 'top')
    
    plt.xlabel('$\mathit{E}$ (V vs $E_{rev}$)') #Cu/$\mathregular{Cu^{2+}}$
    plt.ylabel('$\mathit{\\theta}_\mathrm{Cu}$')
    plt.tick_params(axis='both',which='both',direction='in')
    plt.plot(potential, coverage, color = "dodgerblue", label='Real Data', linewidth = lw, linestyle = "dotted")  #plot it as E vs Cov bec thats how we collect the data
    
    x1 = np.linspace( 0.0001, coverage[-1], 10000) #start, end, # of points
    predicted_potential = frumkin(fitted_params, x1)# potential)  #fits the frumkin isotherm with the least squares fit [g] parameter, and coverage data you have to give potential
    plt.plot(  predicted_potential,  x1, color = "black", label='Fitted Frumkin Isotherm', linestyle =  "solid", linewidth = lw-1) #then plotting predicted E from fitted [g] param vs real coverage
    #plt.legend()
    plt.show()


    # # plot CV vs E_formal
    # plt.figure(figsize=(6,6))
    # plt.subplots_adjust(top=0.96,bottom=0.13, left=0.19, right=0.985, hspace=0.2, wspace=0.2)
    # for i in range(len(frames)): 
    #     frame = frames[SR[i]]
    #     name = psudcap_norm(frame, rates[i], cycle[i], electrode_area)
    #     plt.plot(name['Ewe/V']-E_form, name['C/mF cm^-2'], color=colors[(i)], label = lednames[i], linewidth=lw-1) #mse correction +.3662
    # plt.xlabel('$\mathit{E}$ (V vs $E_{form}$)') #for Cu/Cu^2+, $E_{rev}$, $E_{form}$,  Cu/$\mathregular{Cu^{2+}}$
    # plt.ylabel('   $\mathit{j}$/$\mathit{\\nu}$  (mF/cm\u00b2)') #plt.ylabel('Normalized Current Density (\u03BCF)/cm\u00b2)')
    # plt.axvline(0.0, color = "grey", linestyle = "dashed", linewidth=lw-1) #shows where E_formal is (always 0 on this scale)
    # #plt.axvline(Emin - E_form, color = "red", linestyle = "dashed", linewidth=lw) # shows where the potential for the desired coverage (5%) is on the plot
    # plt.tick_params(axis='both',which='both',direction='in')
    # #plt.legend()#(loc = [.65, 0.1] )  #top right [.75, 0.5] [0,0] is bottom left corner
    # plt.show()

### For this data the starting point cant have a lesser adsorption afterwards bec otherwise youll get a divide by zero error

#print(calculate_formal_potential(pca_25, [0.2, 0.6], 2, SA, R))
def prep_data():
    """ this function is set to get the covergae vs potential range from 0% to the desired covereage set above (should be no greater than 5%)
    first it takes the full region from the onset (0.4) to the end (0.0 V vs E_rev) then it finds the potential at which coverage = 5%.
    it then uses that potential as the new E_min (cathodic scan, min potential is your final bounds) and gets the new data set. that is then 
    passed to the next function for use in fitting"""
    frame = frames[SR[0]]
    
    Emin =  0.0 #goes from 0.4 to 0.2 approx
   
    Emax =  BLpoint + 0.01  #slightly highter than baseline to find the voltage closest
    
    #gets coverage vs poetnial data
    theta_initial, Es_initial = Adsorption_Data(frame, [Emin,Emax], SA, SR, BLpoint, 2)
    #finds the potential at which coverage = desired %
    Emin = find_coverage(theta_initial, Es_initial, coverage)
    #new data at the specific coverage we want
    theta, Es = Adsorption_Data(frame, [Emin,Emax], SA, SR, BLpoint, 2)

    
    return theta, Es, Emin, Emax

def get_frumkin():
    """ To properly fit the adsorption vs. potential data to the Frumkin adsorption isotherm the theta in the exp(g*Θ) term, 
    is approximated to the langmuir isotherm. this is done as one cannot fit the the equation when solved for potential due to how the data is collected
    (where the potential is the independant variable, the value we change to see how coverage changes). as the goal of the fit is to solve for the interacion term (g)
    it is likely appropriate to substidute the Θ in exp(g*Θ) for the langmuir isotherm as that avoids a function of Θ(Θ, E) and just gives Θ(E). 

    it may be possible to nest the frunkim several times in the Θ term before substituting for the langmuir isotherm"""


    """" Fits to find the interaction parameter g (with g<0 being atteactive interactions, and g>0 being repulsive, g=0 is the ideal langmuir non interactive model)
        and also find the formal standard potential, but what it actually fits is the 'distance too the formal standard potential, from the refernce potential 
        (where all my references are vs the reversibly reduction potential of the cell (E_rev))'
        
        so by taking E_rev-E_dist_to_formal you may plot the CVs vs the actual formal standard potential for a solution at standard state and 1 M"""
        
    theta, es, Emin, Emax = prep_data()
    
    # Actual data
    coverage = np.array(theta) + 0.0005 # x data when fitting 
    potential = np.array(es) # y data for fit
    #print(theta25)
    
    
    #finds error/diff in real and fitted data for optimizing
    def residuals(params, x, y):
        
        y_predicted = frumkin(params, x)
        relative_errors = (y - y_predicted) / y
        return np.sum(relative_errors**2)
    
    def least_squares_relative_error(x, y, initial_guess =  [0, 0.0]):
        """
        Performs least squares fitting using relative error optimization.
        """
        result = minimize(residuals, initial_guess, args=(x, y))
        return result.x
    
    fitted_params = least_squares_relative_error(coverage, potential)  #shows just the fitted parameters from fitting [g]
    #predicted_potential = frumkin(fitted_params, coverage)  #fits the frumkin isotherm with the least squares fit [g] parameter, and coverage data you have to give potential

    print('the fitted parameters are [g, E_fudge] =',fitted_params)
    
    g, e = fitted_params
    

    print( 'the fitted parameters are [g] =',  g) #tells you what [g]is
    print( 'the interaction energy is [γ] =',  (g*RT)/(2*1000), "kJ/mol")
    print( 'the potential distance to the formal potential is [E] =',  e)
    
 
    return coverage, potential, fitted_params, Emin, Emax




"""this has my functions"""

def psudcap_norm(frame, rate, cycle, electrode_area):
   
        name = frame[frame['cycle number']==cycle]
        name['J/uA cm^-2'] = name['<I>/mA']*1000/electrode_area
        name.reset_index(drop=True, inplace=True)
        name['C/uF cm^-2'] = name['J/uA cm^-2']/(rate/1000) #capacitance (current normalized to scan rate) is current divided by scan rate
                    ### scanrate/1000 converts mV/s to uA/us from which uA*us=uC and uC/uV=uF
        name['C/mF cm^-2'] = name['C/uF cm^-2']/1000    
        return name

def find_coverage(thetas, Es, coverage): #Q-list is thetas, E -> Es, and coverage is taarget cov (comparing accross scan rates)
   #Q_list is coverage, Es = corrected potential, target coverage (arb) 
    '''This function finds the potential for which a given target coverage is reached'''
    
    for n, theta in enumerate(thetas):
        if theta > coverage:
            print(Es.iloc[n])
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

    frame = frame[frame['cycle number'] ==  cycle]
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

        

plot_fits(frames, SR, cyclenum, legend, colors, SA)