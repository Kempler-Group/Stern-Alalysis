# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:19:23 2025

@author: Owner
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate
import os
import math
from matplotlib.ticker import MaxNLocator

SMALL_SIZE = 20
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
plt.rcParams["legend.loc"] =  'upper left' 
lw = 2 #line width



colors = [    "blue", "red", "navy", "black"]      # 'lightskyblue',    "maroon", "firebrick", "red", "tomato", "lightcoral"
#colors = [ "grey",  "#b87333",   "tomato",  "red",  "firebrick", "maroon"] # color of traces
#colors = colors[::-1] #reverses list

# colors = ["midnightblue", "blue", "dodgerblue"]#, "skyblue"] # color of traces
# colors = colors[::-1] #reverses list

legend = ["10 mM HCl$O_{4}$",  "100 mM HCl$O_{4}$", "1000 mM HCl$O_{4}$"] # $\mathregular{Cu^{2+}}$, HCl$O_{4}$

electrode_areas = [ 0.132, 0.141, 0.132, 0.132, ] #cm^2
Rs = [ 4.946,  39.65, 43, 44.0712 ,  38.4945, ] # resistance, 0 IF CORRECTED ON INSTRUMENT
comp = 0.0 # 15% iR correction

# File loading with dictionary and scan rates
""" there should only be 1 file per dictionary as this analysis takes 1 scan. here we are just averging multiple data sets of 1 scan """
library = [

    {  # Dictionary to store DataFrames and scan rates
     25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\9-1-25 1 mM Cu fresh\CV-25_#015_Au(st006)_Pt_Cu[1mM]_100-1_mM_PCA-Cu_090125_02_CV_C02.mpt",

    },
    {
     
    25: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\H2SO4\100 mM\2025-07-11 1 mM Cu\CV-25-aft_Au(st044)_Pt_RE[1mM]_100_1_HSA_Cu_20250711_02_CV_C02.mpt", 
     
    },
    
    
    
    # {  # Dictionary to store DataFrames and scan rates
    #  500: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\9-1-25 1 mM Cu fresh\CV-500_#015_Au(st006)_Pt_Cu[1mM]_100-1_mM_PCA-Cu_090125_02_CV_C02.mpt",

    # },
    
    # {
    # 5000: r"C:\Users\stern\OneDrive\Documents\School papers\Kempler Lab\data\Au(111)\HClO4\Cu UPD in PCA\100 mM SE\9-1-25 1 mM Cu fresh\CV-5000_#015_Au(st006)_Pt_Cu[1mM]_100-1_mM_PCA-Cu_090125_02_CV_C02.mpt", 
     
    # },

    # ... (rest of your paths)
    ]

potential_correction = [0, 0, 0,  -0.04, 0.0, 0.413, 0.399, 0.381] #if you want to change the potential scale

def multi_normalized (library, electrode_areas):  #rates = [10, 20, 30, 40, 60, 80, 100], rates from scan in frames, SRP = standard reduction potential
    #C_4 = [  "dodgerblue", "blue", "#000000",]#["#501d7a", "#7a51b9", "#9c80e6", "#bdaeef", "#e0d7f7"]

    plt.figure(figsize=(8,8))
    plt.subplots_adjust(top=0.95, bottom=0.095, left=0.13, right=0.965, hspace=0.2, wspace=0.2)

    
    for i in range(len(library)): #iterate through library, each set of data
        ###processing the data to usable form
        head=0
        frames = {}  # Dictionary to store DataFrames, keyed by scan rate
        for sr, file in library[i].items():
            try:
                df = pd.read_csv(file, delimiter='\t', header=head)
                frames[sr] = df  # Store the DataFrame with the scan rate as the key
            except FileNotFoundError:
                print(f"Error: File not found: {file}")
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        scan_rates = list(frames.keys()) # Extract scan rates from the dictionary keys
        cyclenum = [2] * len(library[i])  # Create a list of cycle numbers if needed
        ###
        

        frame = frames[scan_rates[0]]
        name = psudcap_norm(frame, scan_rates[0], cyclenum[0], electrode_areas[i], Rs[i])
        plt.plot(name['Ewe/V'] - potential_correction[i], name['C/mF cm^-2'], color=colors[i], label = legend[i], linewidth=lw) #mse correction +.3662 label = legend[i],
    
    
    #plt.axvline(0.0, color = "black", linestyle = "dashed")
    # plt.axvline(0.2071, color = C_4[0], linestyle = "dashed")
    # plt.axvline(0.1995, color = C_4[1], linestyle = "dashed")
    # plt.axvline( 0.1779, color = C_4[2], linestyle = "dashed")
    # plt.axvline(0.148, color = C_4[0], linestyle = "dashed")
    # plt.axvline(0.111, color = C_4[1], linestyle = "dashed")
    # plt.axvline( 0.060, color = C_4[2], linestyle = "dashed")
    
    plt.xlabel(' $\mathit{E}$ (V vs $E_{rev}$)') #for Cu/Cu^2+, $E_{rev}$, $E_{form}$,  Cu/$\mathregular{Cu^{2+}}$
    plt.ylabel(' $\mathit{j}$/$\mathit{\\nu}$  (mF/cm\u00b2)') #plt.ylabel('Normalized Current Density (\u03BCF)/cm\u00b2)')

    # plt.ylim(-1.5, )
    plt.xlim(-0.01, 0.45)
    #plt.xticks([.25, 0.30, 0.35, 0.40]) 
    plt.tick_params(axis='both',which='both', direction='in', width = 1)
    #plt.legend(frameon = False)
    #plt.title(title)


def psudcap_norm(frame, rate, cycle, electrode_area, R):
        """ takes the frame and gets the iR corrected  "pseidocapacitence"  aka currenet density/scanr rate in mili farads (mF) """
        
        name = frame[frame['cycle number']==cycle]
        #name = name.iloc[round(len(name) / 2):].copy()
        #name = name.iloc[:round(len(name) / 2)].copy() 
        
        name['J/uA cm^-2'] = name['<I>/mA']*1000/electrode_area
        name.reset_index(drop=True, inplace=True)
        name['C/uF cm^-2'] = name['J/uA cm^-2']/(rate/1000) #capacitance (current normalized to scan rate) is current divided by scan rate
                    ### scanrate/1000 converts mV/s to uA/us from which uA*us=uC and uC/uV=uF
        name['E_corr/V'] = name['Ewe/V'] - (name['<I>/mA'] / 1000) * R * comp
        name['C/mF cm^-2'] = name['C/uF cm^-2']/1000    
        
        return name
    


multi_normalized (library, electrode_areas)