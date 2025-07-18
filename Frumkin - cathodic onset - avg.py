# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:25:25 2025

@author: stern
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y label0
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["font.family"] = "Arial" #MESS WITH THIS
plt.rcParams["mathtext.default"] = "regular"  # MAKES ITALICS NORMAL
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
lw = 4 #line width



###start here
conc = 1 # mM conc
concentration = conc/1000 # Molar 10 mM = 0.010 M

electrode_areas = [ 0.132, 0.130, 0.132,] #cm^2
Rs = [48.2021, 57.9279, 60.4307, ] # resistance, 0 IF CORRECTED ON INSTRUMENT
comp = 0.15 # fractional %, iR compensation, if 85% on instrument, do remaining 15% here



BLpoint = 0.43  ###baseline point
Emax =  BLpoint + 0.01  #slightly highter than baseline to find the voltage closest
coverage = 0.05 # 5%, the low coverages for which a linerar approx would work and where we want to compare isotherms at

### Au site density can be back calculated from "ideal" Cu coverage, change for respective ions
TCD = 440e-6 #C/cm^2, Theortical Charge Density (site occupancy)
N = 2 # elementary charge/# of electrons (n = 2 for Cu2+)
mol_Au = TCD / (N * 96485) # C/cm^2, n, C/mol, will need to change 440e-6 (Cu site cov)
# Would be better to derive this using lattice parameters
#print(mol_Au, "mol cm^(-2)")



#colors = ["midnightblue", "blue", "dodgerblue", "skyblue"] # color of dots
colors = [  "lightcoral",   "tomato",  "red",  "firebrick", "maroon"] 

# File loading with dictionary and scan rates
""" there should only be 1 file per dictionary as this analysis takes 1 scan. here we are just averging multiple data sets of 1 scan """
library = [

    {  # Dictionary to store DataFrames and scan rates
      25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\13-1-25 1 mM Cu 1-week\CV-25_#015_Au(st006)_Pt_Cu[1mM]_100-1_mM_PCA-Cu_130125_02_CV_C02.mpt",

    },
    {  # Dictionary to store DataFrames and scan rates
      25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\30-1-25 1 mM Cu fresh -redo\CV-25_Au(st007)_Pt_Cu[1mM]_100-1_mM_PCA-Cu_300125_02_CV_C01.mpt",

    },
    # {  # Dictionary to store DataFrames and scan rates, the scan rates are the keys to the file path
    #   25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\9-1-25 1 mM Cu fresh\CV-25_#015_Au(st006)_Pt_Cu[1mM]_100-1_mM_PCA-Cu_090125_02_CV_C02.mpt",
    
    # },
    # ... (rest of your paths)
    ]



C = coverage
RT = 298.15 * 8.314 #J/mol 
F = 96485 #C/mol
p = RT/F 

def frumkin(params, x): #solved for potential
    g,  e = params
    #concentration = 1
    return -p*(np.log( x / ( 1 - x )) + g*(x-0.5) -np.log(concentration)/N ) + e #g may be backwards here
    # Frumkin: Θ = (K_eq*C_A)/(exp(g*Θ) - K_eq*C_A), Langmuir: Θ = (K_eq*C_A)/(1 - K_eq*C_A) solve for E
    
def plot_avgfits(data_frame, electrode_areas, Rs, colors):  #rates = [10, 20, 30, 40, 60, 80, 100], rates from scan in frames
    
    Emin =  0.0 #goes from 0.4 to 0.2 approx

    # plt.plot(eq) in nice box
    plt.figure(figsize=(8,6.5))
    plt.subplots_adjust( top=0.930, bottom=0.155, left=0.165, right=0.980, hspace=0.2, wspace=0.2)

    gs = []
    es = []
    E_lower = []
    """iterates through each sub dictionary in the library, then for each data set (experimntal trial set)
        it get the interaction paramater (g) and formal potential (e) plots them ..."""
    for i in range(len(data_frame)): #iterate through library, each set of data
        ###processing the data to usable form
        head=0
        frames = {}  # Dictionary to store DataFrames, keyed by scan rate
        for sr, file in data_frame[i].items():
            try:
                df = pd.read_csv(file, delimiter='\t', header=head)
                frames[sr] = df  # Store the DataFrame with the scan rate as the key
            except FileNotFoundError:
                print(f"Error: File not found: {file}")
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        scan_rates = list(frames.keys()) # Extract scan rates from the dictionary keys
        legend = [f"{sr/1000} V/s" for sr in scan_rates]  # Dynamic legend creation
        cyclenum = [2] * len(data_frame[i])  # Create a list of cycle numbers if needed
        ###
        
        theta, Es = prep_data(frames, scan_rates[0], electrode_areas[i], cyclenum[0], Emin, Emax, BLpoint, Rs[i])
        
        fitted_params, coverage, potential, Emin = get_frumkin(theta, Es, Emin, Emax)
        g, e = fitted_params
        plt.plot(potential, coverage, color=colors[i], linestyle = "dotted", linewidth = lw, alpha = 0.5)  #plotting each run, alpha is transparancy
        
        E_lower.append(potential[-1])
        gs.append(g)
        es.append(e)

    g_avg = sum(gs)/len(gs)
    g_stdv = (sum([((x - g_avg) ** 2) for x in gs]) /len(gs))**0.5
    
    e_avg = sum(es)/len(es)
    e_stdv = (sum([((x - e_avg) ** 2) for x in es]) /len(es))**0.5
    
    

    print( 'the interaction parameter is [g] =',  g_avg, ' and the stdev is', g_stdv) #tells you what [g]is
    print( 'the interaction energy is [γ] =',  (g*RT)/(2*1000), 'kJ/mol')
    print( 'the potential distance to the formal potential is [E] =',  e_avg, ' and the stdev is', e_stdv)


    #plot the frumkin fit
    g = g_avg
    e = e_avg

    E_corr = e + (0.059/N) * np.log10(concentration)
    print(E_corr)
    print(e)

    conc = f"conc = {concentration*1000} mM "
    g_text = "$\mathit{g}$"
    interaction_param = f"{g_text} = {g:.2g}±{g_stdv:.1g}"
    gamma =  (g*RT)/(2*1000)
    symbol = "$\mathit{γ}$"
    ef = "$\mathit{E}_{f}$"
    p_m = "\\u+00B1"
    er = "$\mathit{E}_{1/2}$"
    E_rev = "$\mathit{E}_{rev}$"
    formal_pot = f"{ef} = {e:.4f}±{e_stdv:.1g} V vs {E_rev}"
    
    corr_pot = f"{er} = {E_corr:.2g}±{e_stdv:.1g} V " #vs {E_rev}
    
    theta = "$\mathit{\\theta}_\mathrm{Cu}$"
    coverage_str = f" {theta} = {C}"


    interaction_energy = f"{symbol} = {gamma:.1f} kJ/mol"
    

    # Display the equations on the plot
    plt.text(0.45, 0.97, conc, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.text(0.435, 0.90, coverage_str, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.text(0.45, 0.83, interaction_param, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    plt.text(0.45, 0.76, interaction_energy, transform=plt.gca().transAxes, fontsize=24, color='black', verticalalignment='top')
    #plt.text(0.45, 0.83, formal_pot ,transform = plt.gca().transAxes, fontsize=24, verticalalignment = 'top')
    plt.text(0.45, 0.68, corr_pot ,transform = plt.gca().transAxes, fontsize=24, verticalalignment = 'top')

    
    plt.xlabel('$\mathit{E}$ (V vs $E_{rev}$)') #Cu/$\mathregular{Cu^{2+}}$
    plt.ylabel('$\mathit{\\theta}_\mathrm{Cu}$')
    plt.tick_params(axis='both',which='both',direction='in')
    plt.ylim([-0.0025, 0.0525])
    
    fitted_params = [g, e]
    #Emin = sum(E_lower)/len(E_lower)
    x1 = np.linspace( 0.0005, coverage, 10000) #potential start, end, # of points
    predicted_pot = frumkin(fitted_params, x1)# potential)  #fits the frumkin isotherm with the least squares fit [g] parameter, and coverage data you have to give potential
   
    plt.plot( predicted_pot, x1,  color = "black", label='Fitted Frumkin Isotherm', linestyle =  "solid", linewidth = lw-1) #then plotting predicted E from fitted [g] param vs real coverage
    

    #plt.legend()
    plt.show()


### For this data the starting point cant have a lesser adsorption afterwards bec otherwise youll get a divide by zero error
#print(calculate_formal_potential(pca_25, [0.2, 0.6], 2, SA, R))






"""this has my functions"""
def prep_data(frames, SR, SA, cyclenum, Emin, Emax, BLpoint, R):
    """ this function is set to get the covergae vs potential range from 0% to the desired covereage set above (should be no greater than 5%)
    first it takes the full region from the onset (0.4) to the end (0.0 V vs E_rev) then it finds the potential at which coverage = 5%.
    it then uses that potential as the new E_min (cathodic scan, min potential is your final bounds) and gets the new data set. that is then 
    passed to the next function for use in fitting"""
    frame = frames[SR]
    
    #gets coverage vs poetnial data
    theta_initial, Es_initial = Adsorption_Data(frame, [Emin,Emax], SA, SR, BLpoint, cyclenum, R)
    
    #finds the potential at which coverage = desired %
    Emin = find_coverage(theta_initial, Es_initial, coverage)
    #new data at the specific coverage we want
    theta, Es = Adsorption_Data(frame, [Emin,Emax], SA, SR, BLpoint, cyclenum, R)

    return theta, Es

def get_frumkin(theta, es, Emin, Emax):
    """ To properly fit the adsorption vs. potential data to the Frumkin adsorption isotherm the theta in the exp(g*Θ) term, 
    is approximated to the langmuir isotherm. this is done as one cannot fit the the equation when solved for potential due to how the data is collected
    (where the potential is the independant variable, the value we change to see how coverage changes). as the goal of the fit is to solve for the interacion term (g)
    it is likely appropriate to substidute the Θ in exp(g*Θ) for the langmuir isotherm as that avoids a function of Θ(Θ, E) and just gives Θ(E). 

    it may be possible to nest the frunkim several times in the Θ term before substituting for the langmuir isotherm"""


    """" Fits to find the interaction parameter g (with g<0 being atteactive interactions, and g>0 being repulsive, g=0 is the ideal langmuir non interactive model)
        and also find the formal standard potential, but what it actually fits is the 'distance too the formal standard potential, from the refernce potential 
        (where all my references are vs the reversibly reduction potential of the cell (E_rev))'
        
        so by taking E_rev-E_dist_to_formal you may plot the CVs vs the actual formal standard potential for a solution at standard state and 1 M"""
        
    
    #theta, es, Emin, Emax = prep_data()
    
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
    
 
    return fitted_params, coverage, potential,  Emin


def find_coverage(thetas, Es, coverage): #Q-list is thetas, E -> Es, and coverage is taarget cov (comparing accross scan rates)
   #Q_list is coverage, Es = corrected potential, target coverage (arb) 
    '''This function finds the potential for which a given target coverage is reached'''
    
    for n, theta in enumerate(thetas):
        if theta > coverage:
            print(Es.iloc[n])
            return Es.iloc[n]
        else:
            continue
        
def Adsorption_Data(frame, bounds, electrode_area, scanRate, baseLine, cycle, R):  # Added R as a parameter with default
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

    frame = frame[frame['cycle number'] == cycle]
    frame = frame.iloc[:round(len(frame) / 2)].copy()  # get first half of trace, Use .copy() to avoid SettingWithCopyWarning

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
        #intergration (make box with hight J and J+1, and length dt, then divide to get trangle (trapaziod integration bec in a for-loop)
        Q += (corrFrame_J_values[i] + corrFrame_J_values[i - 1]) * dt / 2 
        adsorption = -Q / TCD #normalize to coverage
        Q_list.append(adsorption)
    print(f"the highst coverage is {max(Q_list)*100} %")
    # E has one more element than Q_list, so we slice E to match
    E_matched = E.iloc[1:] # slice E to match the length of Q_list

    return Q_list, E_matched  # Return E_matched

plot_avgfits(library, electrode_areas, Rs, colors)