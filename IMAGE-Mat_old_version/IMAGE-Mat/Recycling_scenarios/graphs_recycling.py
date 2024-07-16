# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2022
@author: deetman@cml.leidenuniv.nl
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd

idx = pd.IndexSlice                     # needed for slicing multi-index

# Great wave of Kanagawa
original_palette = ["#21395a", "#21395a", "#2e58a4", "#b69e71", "#e3ded4", "#71aec7", "#4f5357", "#deab91"]

# plot waste-results
def graph_sectors(electricity, buildings, vehicle, scenario_variant):   
    
    current_palette = [original_palette[2]] + original_palette[2:5]
    mat_select = ['Cu','Concrete','Steel']
    
    plot_start = 2000
    plot_end = 2060
    time = np.array(range(plot_start, plot_end+1))
    
    plt.figure()
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Disaggregation of global E-o-L waste flows, by sector (3 example materials)', y=0.98, fontsize=11)
    plt.tight_layout(pad=3.5)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.25)
    
    ax1.set_title('Electricity', fontsize=10)
    ax1.set_ylabel('Copper (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.stackplot(time, list( - electricity.loc[idx[:,'losses',mat_select[0]],time].sum()), list(electricity.loc[idx[:,'losses',mat_select[0]],time].sum()),  list(electricity.loc[idx[:,'reusable',mat_select[0]],time].sum()), list(electricity.loc[idx[:,'recyclable',mat_select[0]],time].sum()), labels=["losses", "losses", "reusable","recyclable"], colors=current_palette)
    
    ax2.set_title('Buildings', fontsize=10)
    ax2.set_ylabel('Concrete (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.stackplot(time, list( - buildings.loc[idx[:,'losses',mat_select[1]],time].sum()), list(buildings.loc[idx[:,'losses',mat_select[1]],time].sum()), list(buildings.loc[idx[:,'reusable',mat_select[1]],time].sum()), list(buildings.loc[idx[:,'recyclable',mat_select[1]],time].sum()), labels=["losses", "losses", "reusable","recyclable"], colors=current_palette)
    
    ax3.set_title('Vehicles', fontsize=10)
    ax3.set_ylabel('Steel (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.margins(x=0)
    ax3.stackplot(time, list( - vehicle.loc[idx[:,'losses',mat_select[2]],time].sum()), list(vehicle.loc[idx[:,'losses',mat_select[2]],time].sum()), list(vehicle.loc[idx[:,'reusable',mat_select[2]],time].sum()), list(vehicle.loc[idx[:,'recyclable',mat_select[2]],time].sum()), labels=["losses", "losses", "reusable","recyclable"], colors=current_palette)
    
    plt.legend(loc=8, ncol=3, bbox_to_anchor=(-0.8, -0.32), frameon=False, fontsize=10) 
    plt.savefig('Recycling_scenarios\\output\\' + scenario_variant + '3_sectors_waste.jpg', dpi=600)


# check global sectoral inflow
def graph_sectors_inflow(electricity, buildings, vehicle, scenario_variant):   

    
    current_palette = ['limegreen','blue','red']
    mat_select = ['Cu','Aluminium','Steel']
    
    plot_start = 2000
    plot_end = 2060
    time = np.array(range(plot_start, plot_end+1))
    time_str = [str(i) for i in time]
    
    plt.figure()
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Disaggregation of global waste flows, by sector (3 example materials)', y=0.98, fontsize=11)
    plt.tight_layout(pad=3.5)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.25)
    
    ax1.set_title('Copper', fontsize=10)
    ax1.set_ylabel('Copper (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.plot(time, list(electricity.loc[mat_select[0],time_str]), label=["electricity"], color=current_palette[0])
    ax1.plot(time, list(buildings.loc[mat_select[0],time_str]),  label=["building"], color=current_palette[1])
    ax1.plot(time, list(vehicle.loc[mat_select[0],time_str]), label=["vehicle"], color=current_palette[2])
    
    ax2.set_title('Aluminium', fontsize=10)
    ax2.set_ylabel('Aluminium (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.plot(time, list(electricity.loc[mat_select[1],time_str]), label=["electricity"], color=current_palette[0])
    ax2.plot(time, list(buildings.loc[mat_select[1],time_str]),  label=["building"], color=current_palette[1])
    ax2.plot(time, list(vehicle.loc[mat_select[1],time_str]), label=["vehicle"], color=current_palette[2])
     
    ax3.set_title('Steel', fontsize=10)
    ax3.set_ylabel('Steel (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.margins(x=0)
    ax3.plot(time, list(electricity.loc[mat_select[2],time_str]), label=["electricity"], color=current_palette[0])
    ax3.plot(time, list(buildings.loc[mat_select[2],time_str]),  label=["building"], color=current_palette[1])
    ax3.plot(time, list(vehicle.loc[mat_select[2],time_str]), label=["vehicle"], color=current_palette[2])
       
    plt.legend(loc=8, ncol=3, bbox_to_anchor=(-0.8, -0.22), frameon=False, fontsize=10) 
    plt.savefig('Recycling_scenarios\\output\\' + scenario_variant + '3_sectors_inflow_global.jpg', dpi=600)


# input: sum of global inflows (annual material demand in kt/yr)
def graph_inflow(global_demand, scenario_variant, demand_cat):   
    
    current_palette = original_palette[3:7]
    mat_select = ['Cu','Concrete','Steel']
    
    plot_start = 2000
    plot_end = 2060
    time = np.array(range(plot_start, plot_end+1))
    
    plt.figure()
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Disaggregation of global material demand (3 example materials)', y=0.98, fontsize=11)
    plt.tight_layout(pad=3.5)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.25)
        
    ax1.set_title('Copper', fontsize=10)
    ax1.set_ylabel('Copper (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.stackplot(time, list(global_demand.loc[idx['reused',mat_select[0]],time]), list(global_demand.loc[idx['regional recycling',mat_select[0]],time]),  list(global_demand.loc[idx['global recycling',mat_select[0]],time]), list(global_demand.loc[idx['virgin input',mat_select[0]],time]), labels=demand_cat, colors=current_palette)
    
    ax2.set_title('Concrete', fontsize=10)
    ax2.set_ylabel('Concrete (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.stackplot(time, list(global_demand.loc[idx['reused',mat_select[1]],time]), list(global_demand.loc[idx['regional recycling',mat_select[1]],time]),  list(global_demand.loc[idx['global recycling',mat_select[1]],time]), list(global_demand.loc[idx['virgin input',mat_select[1]],time]), labels=demand_cat, colors=current_palette)
    
    ax3.set_title('Steel', fontsize=10)
    ax3.set_ylabel('Steel (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.margins(x=0)
    ax3.stackplot(time, list(global_demand.loc[idx['reused',mat_select[2]],time]), list(global_demand.loc[idx['regional recycling',mat_select[2]],time]),  list(global_demand.loc[idx['global recycling',mat_select[2]],time]), list(global_demand.loc[idx['virgin input',mat_select[2]],time]), labels=demand_cat, colors=current_palette)
    
    plt.legend(loc=8, ncol=4, bbox_to_anchor=(-0.8, -0.22), frameon=False, fontsize=10) 
    plt.savefig('Recycling_scenarios\\output\\' + scenario_variant + 'Global_inflow.jpg', dpi=600)

# input: sum of global waste flows
def graph_waste(global_waste, global_demand, scenario_variant, demand_cat):   

    #global_waste  = regional_waste.sum(level=[1,2])
    #global_demand = regional_demand.sum(level=[1,2])
        
    negative_cat = ['surplus losses', 'losses']
    all_cats     = ['surplus losses'] + negative_cat + demand_cat
    sum_negative = - global_waste.loc[idx[negative_cat,:],:].sum(level=1)
    
    #global_demand = regional_demand.sum(level=[1,2])
    current_palette = original_palette[0:7]
    # current_palette = ['darkgreen', 'darkgreen', 'limegreen', 'blue','red','yellow','slateblue']
    mat_select = ['Cu','Concrete','Steel']
    
    plot_start = 2000
    plot_end = 2060
    time = np.array(range(plot_start, plot_end+1))
    
    plt.figure()
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle('Disaggregation of global material flows (3 example materials)', y=0.98, fontsize=11)
    plt.tight_layout(pad=4)
    plt.subplots_adjust(hspace = 0.25, wspace = 0.35)
        
    ax1.set_title('Copper', fontsize=10)
    ax1.set_ylabel('Copper (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.stackplot(time, list(sum_negative.loc[mat_select[0],time]), list(global_waste.loc[idx['surplus losses',mat_select[0]],time]), list(global_waste.loc[idx['losses',mat_select[0]],time]), list(global_demand.loc[idx['reused',mat_select[0]],time]), list(global_demand.loc[idx['regional recycling',mat_select[0]],time]),  list(global_demand.loc[idx['global recycling',mat_select[0]],time]), list(global_demand.loc[idx['virgin input',mat_select[0]],time]), labels=all_cats, colors=current_palette)
    
    ax2.set_title('Concrete', fontsize=10)
    ax2.set_ylabel('Concrete (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.stackplot(time, list(sum_negative.loc[mat_select[1],time]), list(global_waste.loc[idx['surplus losses',mat_select[1]],time]), list(global_waste.loc[idx['losses',mat_select[1]],time]), list(global_demand.loc[idx['reused',mat_select[1]],time]), list(global_demand.loc[idx['regional recycling',mat_select[1]],time]),  list(global_demand.loc[idx['global recycling',mat_select[1]],time]), list(global_demand.loc[idx['virgin input',mat_select[1]],time]), labels=all_cats, colors=current_palette)
    
    ax3.set_title('Steel', fontsize=10)
    ax3.set_ylabel('Steel (kt/yr)', rotation='vertical', y=0.8, fontsize=10)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.margins(x=0)
    ax3.stackplot(time, list(sum_negative.loc[mat_select[2],time]), list(global_waste.loc[idx['surplus losses',mat_select[2]],time]), list(global_waste.loc[idx['losses',mat_select[2]],time]), list(global_demand.loc[idx['reused',mat_select[2]],time]), list(global_demand.loc[idx['regional recycling',mat_select[2]],time]),  list(global_demand.loc[idx['global recycling',mat_select[2]],time]), list(global_demand.loc[idx['virgin input',mat_select[2]],time]), labels=all_cats, colors=current_palette)
    
    plt.legend(loc=8, ncol=6, bbox_to_anchor=(-0.8, -0.37), frameon=False, fontsize=10) 
    plt.savefig('Recycling_scenarios\\output\\' + scenario_variant + 'Global_flows.jpg', dpi=600)



