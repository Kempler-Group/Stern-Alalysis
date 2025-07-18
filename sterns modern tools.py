# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:04:59 2025

@author: Owner
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math 

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
plt.rcParams["legend.loc"] =  'upper right' 
lw = 2 #line width




electrode_area = 0.142 #cm^2
R = 46.641 # resistance, 0 IF CORRECTED ON INSTRUMENT
comp = 0.15 # fractional %, iR compensation, if 85% on instrument, do remaining 15% here


# bounds for finding min/max current for scanrate dependance 
Emin = 0.015
Emax = Emin + 0.005 

# File loading with dictionary and scan rates
library =       {  # Dictionary to store DataFrames and scan rates, the scan rates are the keys to the file path
    25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Ag UPD\100 mM SE\2025-07-15 10 mM Ag\LSV-25-aft_Au(st045)_Pt_RE[10mM]_100_10_PCA_Ag_20250715_03_LSV_C02.mpt",
    50: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Ag UPD\100 mM SE\2025-07-15 10 mM Ag\LSV-50-v2_Au(st045)_Pt_RE[10mM]_100_10_PCA_Ag_20250715_03_LSV_C02.mpt",
    500: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Ag UPD\100 mM SE\2025-07-15 10 mM Ag\LSV-500-v2_Au(st045)_Pt_RE[10mM]_100_10_PCA_Ag_20250715_03_LSV_C02.mpt",
    1000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Ag UPD\100 mM SE\2025-07-15 10 mM Ag\LSV-1000_Au(st045)_Pt_RE[10mM]_100_10_PCA_Ag_20250715_03_LSV_C02.mpt",
    10000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Ag UPD\100 mM SE\2025-07-15 10 mM Ag\LSV-10000_Au(st045)_Pt_RE[10mM]_100_10_PCA_Ag_20250715_03_LSV_C02.mpt",
    25000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Ag UPD\100 mM SE\2025-07-15 10 mM Ag\LSV-25000_Au(st045)_Pt_RE[10mM]_100_10_PCA_Ag_20250715_03_LSV_C02.mpt",

    # # # #... (rest of your paths)
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



def run_proc(frames):
    

    CV_title = " N/A "#"0.1 mM $\mathregular{Cu^{2+}}$ (100 mM HCl$O_{4}$)" #  Au(111) (100:1:5 mM HCl$O_{4}$:Cu:Cl)
    bounds = [Emin, Emax]
    
    #color = ["#000000", "#3c1713", "#7b241f", "#b42925", "#db4541", "#ee6f6d", "#f79796", "#ffbbba", "#ffdedc", "#ffffff"]
    red = ["#000000", "#1b0a08", "#3c1713", "#5c201a", "#7b241f", "#982722", "#b42925", "#ce2d2b", "#db4541", "#e55b57", "#ee6f6d", "#f48382", "#f79796", "#fca9a8", "#ffbbba", "#ffcdcb", "#ffdedc", "#ffeeec", "#ffffff"]
    
    ### DO NOT CHANGE ANYTHING BELOW THIS LINE (except blanktitle and cycle number ==) !!! ###


    #multi_normalized(frames, CV_title, red) # plots normalized psudocapacitence data #color(3)
    multi_LSV(frames, CV_title, red) # plots normalized psudocapacitence data #color(3)
    #currentrate_dep(frames, bounds, color(2), CV_title) #plots the log scan rate vs current plot for the selected scans, not very useful compared to double log plots below
    #minlogcurrentrate_dep(frames, bounds) #  double log plot of cathodic current vs scanrate (tells scanrate dependance)
    #maxlogcurrentrate_dep(frames, bounds) #  double log plot of anodic current vs scanrate (tells scanrate dependance)



def multi_normalized (frames, title, colors):  #rates = [10, 20, 30, 40, 60, 80, 100], rates from scan in frames
    """ plots the given data frames (CVs) normalized to elecrtode area and scanrate in mF/cm^2.  """

    plt.figure(figsize=(10,10))
    plt.subplots_adjust(top=0.970,bottom=0.09, left=0.125, right=0.99, hspace=0.2, wspace=0.2)
   
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]
        name = psudcap_norm(frame, sr, cyclenum[i], electrode_area)
        plt.plot(name['E_corr/V'], name['C/mF cm^-2'], color=colors[(i)], label = legend [i], linewidth=lw) #mse correction +.3662, 'E_corr/V' 'Ewe/V'
        
    plt.xlabel('$\mathit{E}$ (V vs MSE)') #for Cu/Cu^2+, $E_{rev}$, $E_{form}$,  Cu/$\mathregular{Cu^{2+}}$
    plt.ylabel('   $\mathit{j}$/$\mathit{\\nu}$  (mF/cm\u00b2)') #plt.ylabel('Normalized Current Density (\u03BCF)/cm\u00b2)')
    # #plt.ylim(-0.7, 0.7)
    # plt.xlim(0.0, 0.15)
    
    # plt.ylim(-.5, 0)
    # plt.xlim(0.2, 0.5)
    #plt.axvline(-0.922, color = "grey", linestyle = "dashed")
    #plt.axvline(-1.044, color = "grey", linestyle = "dashed")
    plt.tick_params(axis='both',which='both',direction='in')
    plt.legend(frameon = False)#(loc = [.65, 0.1] )  #top right [.75, 0.5] [0,0] is bottom left corner
    #plt.title(title, size = MEDIUM_SIZE)
    plt.show()
    
def multi_LSV (frames, title, colors):  #rates = [10, 20, 30, 40, 60, 80, 100], rates from scan in frames
    """ plots the given data frames (CVs) normalized to elecrtode area and scanrate in mF/cm^2.  """

    plt.figure(figsize=(10,10))
    plt.subplots_adjust(top=0.970,bottom=0.09, left=0.125, right=0.99, hspace=0.2, wspace=0.2)
   
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]
        name = psudcap_LSV(frame, sr, electrode_area)
        plt.plot(name['E_corr/V'], name['C/mF cm^-2'], color=colors[(i)], label = legend [i], linewidth=lw) #mse correction +.3662, 'E_corr/V' 'Ewe/V'
        
    plt.xlabel('$\mathit{E}$ (V vs $E_{rev}$)') #for Cu/Cu^2+, $E_{rev}$, $E_{form}$,  Cu/$\mathregular{Cu^{2+}}$
    plt.ylabel('   $\mathit{j}$/$\mathit{\\nu}$  (mF/cm\u00b2)') #plt.ylabel('Normalized Current Density (\u03BCF)/cm\u00b2)')
    plt.ylim(-0.01, )
    plt.xlim(0.4, 0.6)
    

    plt.tick_params(axis='both',which='both',direction='in')
    plt.legend(frameon = False)#(loc = [.65, 0.1] )  #top right [.75, 0.5] [0,0] is bottom left corner
    plt.title(title, size = MEDIUM_SIZE)
    plt.show()

    
def minlogcurrentrate_dep(frame, bounds):
    """ takes a list of frames and the potential region you want to find the cathodic scan rate dependance over and plot it on a double log plot.
    the slope of the line tells you about the relation, where a slope of magnitude 1 means a surface controlled rxn and 0.5 is mass transfer controlled
    a slope inbetween is indicitave of a mix, (mayhaps the slow scans are 1 and the fast are 0.5, or other)
    if the magnitude of the slope is greater than 1 then (depending on the system) something weird is going on (maybe an acceleratory mechanism), this is observed fro Cu UPD in HClO4 with Cl- present
    
    the Y-intercept is the psuedocapacitance of the double layer (I believe)"""
    
    
    plt.figure(figsize=(6.5,6.5))
    plt.subplots_adjust(top=0.955, bottom=0.130, left=0.170, right=0.98, hspace=0.2, wspace=0.2)
    
    color = ["#000000", "#061025", "#0a225c", "#052c9c", "#063ac7", "#124eda", "#2862df", "#3b73e4", "#4e83ea", "#6192ef", "#74a0f2", "#87adf3","#9ab9f4", "#acc5f6", "#bed1f7", "#d0ddf9", "#e0e8fb", "#f0f3fd", "#ffffff"]
    
    current = [] #ydata
    lgsc = [] #xdata
    
    
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]
        i_min = mincurrent(frame, bounds, cyclenum[i] , sr, electrode_area, R)
        logsc = math.log10(sr)
        log_mincurrent = -1 * math.log10(abs(i_min))
        current.append(log_mincurrent)
        lgsc.append(logsc)
        plt.plot(logsc, log_mincurrent, 'bo' , markersize = 11, color=color[i % len(color)] , label = legend[i]) #color = colors[i+2]

    # Convert to numpy arrays for easier manipulation
    lgsc = np.array(lgsc)
    current = np.array(current)
    # Check for any potential outliers (simple filtering example)
    mask_min = np.abs(current - np.mean(current)) < 2 * np.std(current)
    # Fit the data
    min_fit_params = np.polyfit(lgsc, current, 1)
    # Generate fit lines
    lgsc_fit = np.linspace(min(lgsc), max(lgsc), 100)
    min_fit_line = np.polyval(min_fit_params, lgsc_fit)
    plt.plot(lgsc_fit, min_fit_line, 'b', label='Min Current Fit')
    # Prepare the equations to display
    min_slope, min_intercept = min_fit_params
    
    E_rev = "$E_{rev}$"
    V = "$\mathit{V}$ "
    min_eq = f"log(min current) = {min_slope:.4f}x + {min_intercept:.4f}" #Min Current: 
    E_range = f"Potential Range: {bounds} {V} vs {E_rev}"
    
    # Display the equations on the plot
    plt.text(0.16, 0.94, min_eq, transform=plt.gca().transAxes, fontsize=18, color='black', verticalalignment='top')
    plt.text(0.16, 0.89, E_range ,transform = plt.gca().transAxes, fontsize=16, verticalalignment = 'top')
    plt.xlabel('log($\mathit{\\nu}$ (mV/s))') 
    plt.ylabel('-log( $\mathit{j}$ (\u03BCA)/cm\u00b2))')
    plt.tick_params(axis='both',which='both',direction='in', width = 1)
    #plt.legend()
    #plt.title(title, size = MEDIUM_SIZE)
    plt.show()

def maxlogcurrentrate_dep(frame, bounds):
    """ takes a list of frames and the potential region you want to find the anodic scan rate dependance over and plot it on a double log plot.
    the slope of the line tells you about the relation, where a slope of magnitude 1 means a surface controlled rxn and 0.5 is mass transfer controlled
    a slope inbetween is indicitave of a mix, (mayhaps the slow scans are 1 and the fast are 0.5, or other)
    if the magnitude of the slope is greater than 1 then (depending on the system) something weird is going on (maybe an acceleratory mechanism), this is observed fro Cu UPD in HClO4 with Cl- present
    
    the Y-intercept is the psuedocapacitance of the double layer (I believe)"""

    plt.figure(figsize=(6.5,6.5))
    plt.subplots_adjust(top=0.955, bottom=0.130, left=0.170, right=0.98, hspace=0.2, wspace=0.2)

    colors = ["#000000", "#1c0b0f", "#3e1923", "#5e2133", "#7e2642", "#9d2850", "#bb265d", "#d6276a", "#ec3177", "#f84d86", "#ff6996", "#ff86a9", "#ff9eba", "#ffb4c9", "#ffc8d7", "#ffdbe4", "#ffecf1", "#ffffff"]    
    
    maxcurr = [] #ydata
    lgsc = [] #xdata
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]
        i_max = maxcurrent(frame, bounds, cyclenum[i] ,  sr, electrode_area, R)
        logsc = math.log10(sr)
        log_maxcurrent = math.log10((i_max))
        maxcurr.append(log_maxcurrent)
        lgsc.append(logsc)
        plt.plot(logsc, log_maxcurrent, 'bo' , markersize = 11, color=colors[i % len(colors)] , label = legend[i]) #color = colors[i+2]
    
    # Convert to numpy arrays for easier manipulation
    lgsc = np.array(lgsc)
    maxcurr = np.array(maxcurr)
    # Check for any potential outliers (simple filtering example)
    mask_max = np.abs(maxcurr - np.mean(maxcurr)) < 2 * np.std(maxcurr)
    # Fit the data
    max_fit_params = np.polyfit(lgsc, maxcurr, 1)
    # Generate fit lines
    lgsc_fit = np.linspace(min(lgsc), max(lgsc), 100)
    max_fit_line = np.polyval(max_fit_params, lgsc_fit)
    plt.plot(lgsc_fit, max_fit_line, 'crimson', label='Max Current Fit')
    
    # Prepare the equations to display
    max_slope, max_intercept = max_fit_params
    
    E_rev = "$E_{rev}$"
    V = "$\mathit{V}$ "
    max_eq = f"log(max current) = {max_slope:.4f}x + {max_intercept:.4f}"
    E_range = f"Potential Range: {bounds} {V} vs {E_rev}"
    # Display the equations on the plot
    plt.text(0.05, 0.95, max_eq, transform=plt.gca().transAxes, fontsize=18, color='black', verticalalignment='top')
    plt.text(0.05, 0.89, E_range ,transform = plt.gca().transAxes, fontsize=16, verticalalignment = 'top')
    plt.xlabel('log($\mathit{\\nu}$ (mV/s))') 
    plt.ylabel('log( $\mathit{j}$ (\u03BCA)/cm\u00b2))')
    plt.tick_params(axis='both',which='both',direction='in', width = 1)
    #plt.legend()
    #plt.title(title, size = MEDIUM_SIZE)
    plt.show()

def currentrate_dep(frame, bounds, colors, title):
    """ *not optimally written
    plots the cathodic and anoic current for a given potential vs the log of the scanrat. not as useful as the double log plot min/maxlogcurrentrate_dep functions
    (takes a range, bounds,  bec there may not be a data point at a specific potential so finds max/min within that range)
    """
    
    
    plt.figure(figsize=(9,9))
    plt.subplots_adjust(top=0.93, bottom=0.09, left=0.17, right=0.985, hspace=0.2, wspace=0.2)
    current = [] #ydata
    maxcurr = []
    lgsc = [] #xdata
    
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]

    
        i_min = mincurrent(frame, bounds, cyclenum[i] , sr, electrode_area, R)
        i_max = maxcurrent(frame, bounds, cyclenum[i] , sr, electrode_area, R)
        logsc = math.log10(sr)
        current.append(i_min) #cathodic current
        maxcurr.append(i_max) #anodic current
        lgsc.append(logsc)
        plt.plot(logsc, i_min, 'bo' , color=colors[i % len(colors)] , label = legend[i]) #color = colors[i+2]
        #plt.plot(logsc, i_max, 'bo' , color=colors[i % len(colors)])
    #print(current)
    #print(maxcurr)
    
    plt.xlabel('Log(scanrate(mV/s))') 
    plt.ylabel('Current Density (\u03BCJ)/cm\u00b2)')
    plt.tick_params(axis='both',which='both',direction='in')
    plt.legend()
    plt.title("Cathodic Current")
    plt.show()
    
    plt.figure(figsize=(9,9))
    plt.subplots_adjust(top=0.93, bottom=0.09, left=0.17, right=0.985, hspace=0.2, wspace=0.2)
    current = [] 
    maxcurr = [] #ydata
    lgsc = [] #xdata
    for i, sr in enumerate(scan_rates):  # Iterate through scan rates
        frame = frames[sr]
        i_min = mincurrent(frame, bounds, cyclenum[i] ,  sr, electrode_area, R)
        i_max = maxcurrent(frame, bounds, cyclenum[i] ,  sr, electrode_area, R)
        logsc = math.log10(sr)
        current.append(i_min) #cathodic current
        maxcurr.append(i_max) #anodic current
        lgsc.append(logsc)
        #plt.plot(logsc, i_min, 'bo' , color=colors[i % len(colors)] , label = labels[i]) #color = colors[i+2]
        plt.plot(logsc, i_max, 'bo' , color=colors[i % len(colors)], label = legend[i])
    #print(current)
    #print(maxcurr)
    
    plt.xlabel('Log(scanrate(mV/s))') 
    plt.ylabel('Current Density (\u03BCA)/cm\u00b2)')
    plt.tick_params(axis='both',which='both',direction='in')
    plt.legend()
    plt.title("Anodic Current")
    plt.show()

def psudcap_norm(frame, rate, cycle, electrode_area):
        """ takes the frame and gets the iR corrected  "pseidocapacitence"  aka currenet density/scanr rate in mili farads (mF) """
        
        name = frame[frame['cycle number'] == cycle]
        name['J/uA cm^-2'] = name['<I>/mA']*1000/electrode_area
        name.reset_index(drop=True, inplace=True)
        name['C/uF cm^-2'] = name['J/uA cm^-2']/(rate/1000) #capacitance (current normalized to scan rate) is current divided by scan rate
                    ### scanrate/1000 converts mV/s to uA/us from which uA*us=uC and uC/uV=uF
        # Correct potential for IR drop (vectorized)
        name['E_corr/V'] = name['Ewe/V'] - (name['<I>/mA'] / 1000) * R * comp
        name['C/mF cm^-2'] = name['C/uF cm^-2']/1000    
        
        return name
    
    
def psudcap_LSV(frame, rate, electrode_area):
        """ takes the frame and gets the iR corrected  "pseidocapacitence"  aka currenet density/scanr rate in mili farads (mF) """
        
        name = frame
        name['J/uA cm^-2'] = name['<I>/mA']*1000/electrode_area
        name.reset_index(drop=True, inplace=True)
        name['C/uF cm^-2'] = name['J/uA cm^-2']/(rate/1000) #capacitance (current normalized to scan rate) is current divided by scan rate
                    ### scanrate/1000 converts mV/s to uA/us from which uA*us=uC and uC/uV=uF
        # Correct potential for IR drop (vectorized)
        name['E_corr/V'] = frame['Ewe/V'] - (frame['<I>/mA'] / 1000) * R * comp
        name['C/mF cm^-2'] = name['C/uF cm^-2']/1000    
        
        return name

def current_density(frame, cycle, electrode_area, R):
    """" takes the frame and gets the raw iR corrected currenet density  in micro amps (uA) """
    
    name = frame[frame['cycle number']==cycle]
    name['J/uA cm^-2'] = name['<I>/mA']*1000/electrode_area
    frame['Ewe/V'] = frame['Ewe/V'] - ((frame['<I>/mA']/1000*R*comp))
    name.reset_index(drop=True, inplace=True)
    
    return name

def color(thing): #enter in color(#) to call the color gradient from list: col
    #magenta/pink 6-7 usable
    pink = ["#000000", "#31111d", "#661a3a", "#991352", "#bd296b", "#d34e82", "#e26f98", "#ee8ead", "#f8abc1", "#fdc7d6", "#ffe3eb"]
    #green 12 usable
    green = ["#000000", "#0f140e", "#1f2e1f", "#2c442c", "#355936", "#3c6e3f", "#418146", "#44944d", "#45a753", "#43ba57", "#3fcc5b", "#37dd5e", "#44ed69", "#6df881", "#9bffa1", "#d2ffcf", "#ffffff"]
    #blue 12 usable
    blue = ["#000000", "#061025", "#0a225c", "#052c9c", "#063ac7", "#124eda", "#2862df", "#3b73e4", "#4e83ea", "#6192ef", "#74a0f2", "#87adf3","#9ab9f4", "#acc5f6", "#bed1f7", "#d0ddf9", "#e0e8fb", "#f0f3fd", "#ffffff"]
    #red12 usable
    red = ["#000000", "#1b0a08", "#3c1713", "#5c201a", "#7b241f", "#982722", "#b42925", "#ce2d2b", "#db4541", "#e55b57", "#ee6f6d", "#f48382", "#f79796", "#fca9a8", "#ffbbba", "#ffcdcb", "#ffdedc", "#ffeeec", "#ffffff"]
    #red - short
    cred_short = ["#000000", "#3c1713", "#7b241f", "#b42925", "#db4541", "#ee6f6d", "#f79796", "#ffbbba", "#ffdedc", "#ffffff"]
    #4 distinct
    C_4 = ["#000000", "#000000", "black", "dodgerblue", "crimson", "goldenrod"]
    
    col = [ pink, green, blue, red, cred_short, C_4]

    return col[thing]

def mincurrent(dataframe, bounds, cycle, scanrate, SA, R):
    """ finds the minimum current within a given potential range (global or local), if you want a local min then you must still give a potential range
    as there is the risk of not having a data point at a specific potential so this just finds the closest to it within the range
    
    what CV you want, 
    a narrow range [Emin, Emax] (picks lowest point in range), 
    cycle #, 
    scanrate legend title, 
    electrode surface area, 
    resistance (if no bkgn correction on instrument)"""
    
    frame = dataframe[dataframe['cycle number'] == cycle]  #which cycle you want to analyze
    frame['J/uA cm^-2'] = frame['<I>/mA']*1000/SA              # convert to micro amps
    #frame['C/uF cm^-2'] = frame['J/uA cm^-2']/scanrate
    frame['Ewe/V'] = frame['Ewe/V'] - ((frame['<I>/mA']/1000*R*0.85))
    frame = frame[(frame['Ewe/V'] < max(bounds)) & (frame['Ewe/V'] > min(bounds))] #range for both potnetial and current, will convert to 'I/uA' in for loop

    frame.reset_index(drop=True, inplace=True)      #not sure why but doesnt work without
    framemin_index = np.argmin(frame['J/uA cm^-2'])               # slcies files to around desired peak area (one point)
        
    return frame['J/uA cm^-2'].iloc[framemin_index]

def maxcurrent(dataframe, bounds, cycle, scanrate, SA, R):
    """ finds the maximum current within a given potential range (global or local), if you want a local max then you must still give a potential range
    as there is the risk of not having a data point at a specific potential so this just finds the closest to it within the range
    
    what CV you want, 
    a narrow range [Emin, Emax] (picks highest point in range), 
    cycle #, 
    scanrate legend title, 
    electrode surface area, 
    resistance (if no bkgn correction on instrument)"""
    
    frame = dataframe[dataframe['cycle number'] == cycle]  #which cycle you want to analyze
    frame['J/uA cm^-2'] = frame['<I>/mA']*1000/SA              # convert to micro amps
    #frame['C/uF cm^-2'] = frame['J/uA cm^-2']/scanrate
    frame['Ewe/V'] = frame['Ewe/V'] - ((frame['<I>/mA']/1000*R*0.85))
    frame = frame[(frame['Ewe/V'] < max(bounds)) & (frame['Ewe/V'] > min(bounds))] #range for both potnetial and current, will convert to 'I/uA' in for loop

    frame.reset_index(drop=True, inplace=True)      #not sure why but doesnt work without
    framemin_index = np.argmax(frame['J/uA cm^-2'])               # slcies files to around desired peak area (one point)
        
    return frame['J/uA cm^-2'].iloc[framemin_index]


run_proc(frames)