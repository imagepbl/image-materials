"""
Plots on regional storage demand & availability for 6 different regions 
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd

current_palette = sns.color_palette('Paired')
font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 7}


def graph_global_ev_capacity(BEV_dynamic_capacity, PHEV_dynamic_capacity,  max_capacity_BEV, max_capacity_PHEV, variant, sa_settings):

   """
   Plot to compare total battery capacity of cars to the fraction available for v2g purposes
   """

   plt.rc('font', **font)
   
   # Setting up a comparison between stock, inflow & outflow 
   year_select = list(range(2018,2060))
   plotvar = pd.DataFrame(index=year_select, columns=[0,1,2,3])
   plotvar[0] = BEV_dynamic_capacity[year_select].to_numpy()      
   plotvar[1] = PHEV_dynamic_capacity[year_select].to_numpy()     
   plotvar[2] = max_capacity_BEV[year_select].to_numpy()          
   plotvar[3] = max_capacity_PHEV[year_select].to_numpy()          
   
   # Drawing the plot comparing Chinese stock, inflow & outflow 
   fig, ax1 = plt.subplots()
   fig.set_size_inches(12, 9)
   ax2 = ax1.twinx()
   ax1.margins(x=0)
   ax2.margins(x=0)
   ax1.set_ylim(ymin=0, ymax=140)
   ax2.set_ylim(ymin=0, ymax=0.2)
   ax1.plot(plotvar[0] , color='black', linewidth=3.3, label="Battery Electric Vehicle")
   ax1.plot(plotvar[1] , color='darkblue', linewidth=3.3, label="Plugin Hybrid Electric Vehicle")
   ax2.plot(plotvar[2] , '--', color='orange', linewidth=3.3, label="V2G availability BEV")
   ax2.plot(plotvar[3] , '--', color='red', linewidth=3.3, label="V2G availability PHEV")
   
   ax1.set_xlabel('years', fontsize=14)
   ax1.set_ylabel('Average battery capacity per car (in kWh)', color='black', fontsize=18)
   ax2.set_ylabel('Maximum available fraction', color='red', fontsize=18)
   
   ax1.tick_params(axis='both', labelsize=14)
   ax2.tick_params(labelsize=14, colors='red')
   
   #ax1.yaxis.set_label_coords(-0.12,0.83)
   #ax2.yaxis.set_label_coords(1.1,0.83)
   
   ax1.legend(loc='upper left', prop={'size': 18}, borderaxespad=0., frameon=False)
   ax2.legend(loc='lower right', prop={'size': 18}, borderaxespad=0., frameon=False)
   fig.savefig('output\\' + variant + '\\' + sa_settings + '\\graphs\\Global_EV_capacity_per_car.png')



def graph_regional_storage(region_list, storage_PHEV, storage_BEV, storage, variant, scenario, sa_settings):   
   # PLot pannel
   americas        = region_list[0:6]
   europe          = region_list[10:12]
   africa_and_me   = region_list[6:10] + region_list[13:17] + [region_list[25]]
   china_india     = [region_list[17],region_list[19]]
   japan_australia = region_list[22:24]
   rest_of_asia    = [region_list[18]] + [region_list[20]] + [region_list[21]] + [region_list[24]]
   
   plt.rc('font', **font)
   
   if variant == 'BL':
       ylim = [6000,6000,6000,1600,1600,1600]
   else:
       ylim = [12000,12000,12000,8000,8000,8000]
   
   plot_start = 2010
   plot_end = 2060
   
   plt.figure()
   fig, ((ax3, ax1, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(12,8))
   fig.suptitle('Regional grid storage demand vs vehicle storage availability in GWh (' + scenario + ' ' + variant + ')', y=0.95, fontsize=11)
   plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
   
   ax1.set_title('Americas', fontsize=8)
   ax1.set_ylim(0,ylim[1])
   ax1.yaxis.set_ticks_position('left')
   ax1.xaxis.set_ticks_position('bottom')
   ax1.margins(x=0)
   ax1.stackplot(np.array(storage_PHEV.loc[plot_start:plot_end].index), storage_PHEV[americas].loc[plot_start:plot_end].sum(axis=1)/1000, storage_BEV[americas].loc[plot_start:plot_end].sum(axis=1)/1000, labels=["PHEV storage (10% vehicle2grid)","BEV storage (20% vehicle2grid)"], colors=current_palette)
   ax1.plot(np.array(storage.loc[plot_start:plot_end].index), storage[americas].loc[plot_start:plot_end].sum(axis=1)/1000, 'r', label='grid storage demand', linewidth=2)
   
   ax2.set_title('Japan & Australia', fontsize=8)
   ax2.set_ylim(0,ylim[3])
   ax2.set_ylabel('GWh', rotation='vertical', y=0.8, fontsize=7)
   ax2.yaxis.set_ticks_position('left')
   ax2.xaxis.set_ticks_position('bottom')
   ax2.margins(x=0)
   ax2.stackplot(np.array(storage_PHEV.loc[plot_start:plot_end].index), storage_PHEV[japan_australia].loc[plot_start:plot_end].sum(axis=1)/1000, storage_BEV[japan_australia].loc[plot_start:plot_end].sum(axis=1)/1000, labels=["PHEV storage (10% vehicle2grid)","BEV storage (20% vehicle2grid)"], colors=current_palette)
   ax2.plot(np.array(storage.loc[plot_start:plot_end].index), storage[japan_australia].loc[plot_start:plot_end].sum(axis=1)/1000, 'r',  label='grid storage demand', linewidth=2)
   
   ax3.set_title('Europe', fontsize=8)
   ax3.set_ylim(0,ylim[0])
   ax3.set_ylabel('GWh', rotation='vertical', y=0.8, fontsize=7)
   ax3.yaxis.set_ticks_position('left')
   ax3.xaxis.set_ticks_position('bottom')
   ax3.margins(x=0)
   ax3.stackplot(np.array(storage_PHEV.loc[plot_start:plot_end].index), storage_PHEV[europe].loc[plot_start:plot_end].sum(axis=1)/1000, storage_BEV[europe].loc[plot_start:plot_end].sum(axis=1)/1000, labels=["PHEV storage (10% vehicle2grid)","BEV storage (20% vehicle2grid)"], colors=current_palette)
   ax3.plot(np.array(storage.loc[plot_start:plot_end].index), storage[europe].loc[plot_start:plot_end].sum(axis=1)/1000, 'r',  label='grid storage demand', linewidth=2)
   
   ax4.set_title('Africa & Middle East', fontsize=8)
   ax4.set_ylim(0,ylim[4])
   ax4.yaxis.set_ticks_position('left')
   ax4.xaxis.set_ticks_position('bottom')
   ax4.margins(x=0)
   ax4.stackplot(np.array(storage_PHEV.loc[plot_start:plot_end].index), storage_PHEV[africa_and_me].loc[plot_start:plot_end].sum(axis=1)/1000, storage_BEV[africa_and_me].loc[plot_start:plot_end].sum(axis=1)/1000, labels=["PHEV storage (10% vehicle2grid)","BEV storage (20% vehicle2grid)"], colors=current_palette)
   ax4.plot(np.array(storage.loc[plot_start:plot_end].index), storage[africa_and_me].loc[plot_start:plot_end].sum(axis=1)/1000, 'r', label='grid storage demand', linewidth=2)
   
   ax5.set_title('China & India', fontsize=8)
   ax5.set_ylim(0,ylim[2])
   ax5.set_ylabel('GWh', rotation='vertical', y=0.8, fontsize=7)
   ax5.yaxis.set_ticks_position('left')
   ax5.xaxis.set_ticks_position('bottom')
   ax5.margins(x=0)
   ax5.stackplot(np.array(storage_PHEV.loc[plot_start:plot_end].index), storage_PHEV[china_india].loc[plot_start:plot_end].sum(axis=1)/1000, storage_BEV[china_india].loc[plot_start:plot_end].sum(axis=1)/1000, labels=["PHEV storage (10% vehicle2grid)","BEV storage (20% vehicle2grid)"], colors=current_palette)
   ax5.plot(np.array(storage.loc[plot_start:plot_end].index), storage[china_india].loc[plot_start:plot_end].sum(axis=1)/1000, 'r',  label='grid storage demand', linewidth=2)
   
   ax6.set_title('Rest of Asia', fontsize=8)
   ax6.set_ylim(0,ylim[5])
   ax6.yaxis.set_ticks_position('left')
   ax6.xaxis.set_ticks_position('bottom')
   ax6.margins(x=0)
   ax6.stackplot(np.array(storage_PHEV.loc[plot_start:plot_end].index), storage_PHEV[rest_of_asia].loc[plot_start:plot_end].sum(axis=1)/1000, storage_BEV[rest_of_asia].loc[plot_start:plot_end].sum(axis=1)/1000, labels=["PHEV storage (10% vehicle2grid)","BEV storage (20% vehicle2grid)"], colors=current_palette)
   ax6.plot(np.array(storage.loc[plot_start:plot_end].index), storage[rest_of_asia].loc[plot_start:plot_end].sum(axis=1)/1000, 'r', label='grid storage demand', linewidth=2)
   
   plt.legend(loc=8, bbox_to_anchor=(-.1, -0.5), ncol=3, frameon=False, fontsize=8)
   plt.savefig('output\\' + variant + '\\' + sa_settings + '\\graphs\\regional_EV_storage.jpg', dpi=600)


#%% Figure on the storage demand versus the availability in 3 different storage types (phs, evs and dedicated) 
def graph_storage_demand_vs_availability(region_list, phs_storage_theoretical, storage_vehicles, storage, variant, sa_settings):
   # prepare chart variables (appropriate averages & regions)
   north_america   = region_list[0:2]
   europe          = region_list[10:12]
   china           = region_list[19]
   japan_australia = region_list[22:24]
   rest            = region_list[2:10] + region_list[13:17] + [region_list[17]] +  [region_list[18]] + [region_list[20]] + [region_list[21]] + region_list[24:25]
       
   divby = 1000 # 1000 for GWh
    
   plotvar1 = [phs_storage_theoretical.loc[2030,north_america].sum()/divby, phs_storage_theoretical.loc[2030,europe].sum()/divby, phs_storage_theoretical.loc[2030,china]/divby, phs_storage_theoretical.loc[2030,japan_australia].sum()/divby, phs_storage_theoretical.loc[2030,rest].sum()/divby]
   plotvar2 = [phs_storage_theoretical.loc[2040,north_america].sum()/divby, phs_storage_theoretical.loc[2040,europe].sum()/divby, phs_storage_theoretical.loc[2040,china]/divby, phs_storage_theoretical.loc[2040,japan_australia].sum()/divby, phs_storage_theoretical.loc[2040,rest].sum()/divby]
   plotvar3 = [phs_storage_theoretical.loc[2050,north_america].sum()/divby, phs_storage_theoretical.loc[2050,europe].sum()/divby, phs_storage_theoretical.loc[2050,china]/divby, phs_storage_theoretical.loc[2050,japan_australia].sum()/divby, phs_storage_theoretical.loc[2050,rest].sum()/divby]
    
   plotvar4 = [storage_vehicles.loc[2030,north_america].sum()/divby, storage_vehicles.loc[2030,europe].sum()/divby, storage_vehicles.loc[2030,china]/divby, storage_vehicles.loc[2030,japan_australia].sum()/divby, storage_vehicles.loc[2030,rest].sum()/divby]    
   plotvar5 = [storage_vehicles.loc[2040,north_america].sum()/divby, storage_vehicles.loc[2040,europe].sum()/divby, storage_vehicles.loc[2040,china]/divby, storage_vehicles.loc[2040,japan_australia].sum()/divby, storage_vehicles.loc[2040,rest].sum()/divby]
   plotvar6 = [storage_vehicles.loc[2050,north_america].sum()/divby, storage_vehicles.loc[2050,europe].sum()/divby, storage_vehicles.loc[2050,china]/divby, storage_vehicles.loc[2050,japan_australia].sum()/divby, storage_vehicles.loc[2050,rest].sum()/divby]
   
   plotvar7 = [storage.loc[2030,north_america].sum()/divby, storage.loc[2030,europe].sum()/divby, storage.loc[2030,china]/divby, storage.loc[2030,japan_australia].sum()/divby, storage.loc[2030,rest].sum()/divby]
   plotvar8 = [storage.loc[2040,north_america].sum()/divby, storage.loc[2040,europe].sum()/divby, storage.loc[2040,china]/divby, storage.loc[2040,japan_australia].sum()/divby, storage.loc[2040,rest].sum()/divby]
   plotvar9 = [storage.loc[2050,north_america].sum()/divby, storage.loc[2050,europe].sum()/divby, storage.loc[2050,china]/divby, storage.loc[2050,japan_australia].sum()/divby, storage.loc[2050,rest].sum()/divby]
   
   barWidth = 0.25
   r1 = np.arange(5)
   r2 = [x + barWidth for x in r1]
   r3 = [x + barWidth for x in r2]
   
   plt.figure(figsize=(12, 10))
   fig, ax = plt.subplots(figsize=(12,10))
   plt.rc('font', **font)
   plt.yticks(fontsize=16)
   if variant == 'BL':
       ax.set_ylim(ymin=0, ymax=4000)
   else:
       ax.set_ylim(ymin=0, ymax=8000)   
   ax.bar(r1, plotvar1, color='#321b96', width=barWidth, edgecolor='white', label='2030')
   ax.bar(r2, plotvar2, color='#321b96', width=barWidth, edgecolor='white', label='2040')
   ax.bar(r3, plotvar3, color='#321b96', width=barWidth, edgecolor='white', label='2050')
   ax.bar(r1, plotvar4, color='#1b9667', width=barWidth, bottom=plotvar1, edgecolor='white', label='2030')
   ax.bar(r2, plotvar5, color='#1b9667', width=barWidth, bottom=plotvar2, edgecolor='white', label='2040')
   ax.bar(r3, plotvar6, color='#1b9667', width=barWidth, bottom=plotvar3, edgecolor='white', label='2050')
   ax.plot(r1, plotvar7, 'o', color='red', markersize=12)
   ax.plot(r2, plotvar8, 'o', color='red', markersize=12)
   ax.plot(r3, plotvar9, 'o', color='red', markersize=12)
   
   # Add xticks on the middle of the group bars
   plt.xticks([r + barWidth for r in range(5)], ['N.America', 'Europe', 'China', 'Japan&Australia', 'RoW'], fontsize=16)
   plt.ylabel('storage capacity (GWh)', fontsize=18)
   plt.title('storage capacity (pumped hydro and electric vehicles) vs. storage demand in 2030, 2040 & 2050', y=1.08, fontsize=16)
   
   # Legend & text
   legend_elements = [matplotlib.patches.Patch(facecolor='#1b9667', label='Electric Vehicles'),
                      matplotlib.patches.Patch(facecolor='#321b96', label='Pumped Hydro'),
                      matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=14, label='storage demand')]
   plt.legend(handles=legend_elements, loc=8, bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=16)
   
   plt.text(-0.05, plotvar1[0] + plotvar4[0] + 30 ,'2030',fontsize=16,rotation=90)
   plt.text(0.20,  plotvar2[0] + plotvar5[0] + 30 ,'2040',fontsize=16,rotation=90)
   plt.text(0.45,  plotvar3[0] + plotvar6[0] + 30 ,'2050',fontsize=16,rotation=90)
   
   plt.show()
   fig.savefig('output\\' + variant + '\\' + sa_settings + '\\graphs\\regional_storcapcompare.png', dpi=600)



# Figure showing the regional electricity storage demand requirements
def graph_regional_dedicated_storage(oth_storage, variant, scenario, sa_settings):
   """
   Figure with the remaining demand for dedicated (stationary) storage capacity after deployment of V2G
   """ 
      # group regions with less than 2% cumulative capacity in 'other' region
   cumulative_capacity = oth_storage.loc[2015:2060].sum(axis = 0)
   plot_regions = cumulative_capacity[cumulative_capacity.values > (cumulative_capacity.sum()*0.015)]
   regions_rest = list(set(oth_storage.columns) - set(plot_regions.index))
   plotvar = oth_storage.loc[2015:2060, plot_regions.index].transpose().append(pd.Series(oth_storage.loc[2015:2060,regions_rest].sum(axis=1), name='Other')).transpose()
   
   # re-order to have US bump on top
   if 'US' in list(plotvar.columns):
       plotvar  = plotvar[  [ col for col in plotvar.columns if col != 'US' ] + ['US'] ]
   
   #plot figure & save
   plt.figure(figsize=(10, 8))
   plt.stackplot(np.array(plotvar.index), plotvar.values.transpose() / 1000, labels=plotvar.columns, colors=current_palette) # div by 1000 to yield GWh
   plt.margins(0)
   plt.legend(loc=2, frameon=False, fontsize=14)
   plt.xticks(fontsize=18, rotation=25)
   plt.yticks(fontsize=18, rotation=0)
   plt.ylim(top=800) 
   plt.ylabel('Storage capacity (GWh)', fontsize=20)
   plt.xlabel('time (in years)', fontsize=18)
   plt.title('Dedicated storage demand by region (' + scenario + ' ' + variant + ')', fontsize=20)
   plt.savefig('output\\' + variant + '\\' + sa_settings + '\\graphs\\dedicated_storage_capacity_by_region.jpg', dpi=600)


#Figure on storage shares, annual development of the market share (stacked area plot @ 100%)
def graph_market_share(storage_market_share, variant, sa_settings):
   
   lion_tech   = storage_market_share.columns[5:10]    # Lithium-ion technologies (including lifepo4 & LTO)
   flow_tech   = storage_market_share.columns[10:12]   # flow batteries: Zinc-Bromide & Vanadium Redox
   salt_tech   = storage_market_share.columns[12:14]   # molten salt batteries: Sodium sulfur & ZEBRA
   advl_tech   = storage_market_share.columns[14:18]   # Advanced lithium technologies (Li-S, Li-ceramic & Li-air)
   othr_tech   = storage_market_share.columns[3:5]     # NiMH & Lead-Acid
   
   legend_elements3 = ['Flywheels','Compressed Air', 'Lithium-ion', 'Advanced Li', 'Flow Batteries', 'Molten Salt batteries', 'Hydrogen', 'Other']
   color_set = ("#a82c41", "#6a26d1", "#30942b", "#5c56d6", "#edce77", "#f0ba3c", "#babab8", "#4287f5")
   legend_elements3.reverse()
   
   y = np.vstack([np.array(storage_market_share['Flywheel'].astype(float)), np.array(storage_market_share['Compressed Air'].astype(float)), np.array(storage_market_share[lion_tech].sum(axis=1).astype(float)), np.array(storage_market_share[advl_tech].sum(axis=1).astype(float)), np.array(storage_market_share[flow_tech].sum(axis=1).astype(float)), np.array(storage_market_share[salt_tech].sum(axis=1).astype(float)),  np.array(storage_market_share['Hydrogen FC'].astype(float)), np.array(storage_market_share[othr_tech].sum(axis=1).astype(float))])
   fig, ax = plt.subplots(figsize=(14, 10))
   ax.margins(0)
   ax.tick_params(labelsize=16)
   ax.stackplot(np.array(storage_market_share.index[19:90]), np.flip(y[:,19:90], axis=0), labels=legend_elements3, colors=color_set)
   ax.legend(loc='upper left')
   plt.legend(legend_elements3, loc=8, bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=14)
   plt.tight_layout(pad=10)
   plt.savefig('output\\' + variant + '\\' + sa_settings + '\\graphs\\market_share_storage.jpg', dpi=600)

#%% Pie plot of market shares of dedicated electricity storage (batteries etc.)
def graph_market_share_pie(storage_market_share, variant, sa_settings):
   
   lion_tech   = storage_market_share.columns[5:10]    # Lithium-ion technologies (including lifepo4 & LTO)
   flow_tech   = storage_market_share.columns[10:12]   # flow batteries: Zinc-Bromide & Vanadium Redox
   salt_tech   = storage_market_share.columns[12:14]   # molten salt batteries: Sodium sulfur & ZEBRA
   advl_tech   = storage_market_share.columns[14:18]   # Advanced lithium technologies (Li-S, Li-ceramic & Li-air)
   othr_tech   = storage_market_share.columns[3:5]     # NiMH & Lead-Acid
   
   plotvar1 = [storage_market_share.loc[2018,'Flywheel'], storage_market_share.loc[2018,'Compressed Air'], storage_market_share.loc[2018,lion_tech].sum(), storage_market_share.loc[2018,advl_tech].sum(), storage_market_share.loc[2018,flow_tech].sum(), storage_market_share.loc[2018,salt_tech].sum(), storage_market_share.loc[2018,'Hydrogen FC'], storage_market_share.loc[2018,othr_tech].sum()]
   plotvar2 = [storage_market_share.loc[2030,'Flywheel'], storage_market_share.loc[2030,'Compressed Air'], storage_market_share.loc[2030,lion_tech].sum(), storage_market_share.loc[2030,advl_tech].sum(), storage_market_share.loc[2030,flow_tech].sum(), storage_market_share.loc[2030,salt_tech].sum(), storage_market_share.loc[2030,'Hydrogen FC'], storage_market_share.loc[2030,othr_tech].sum()]
   plotvar3 = [storage_market_share.loc[2050,'Flywheel'], storage_market_share.loc[2050,'Compressed Air'], storage_market_share.loc[2050,lion_tech].sum(), storage_market_share.loc[2050,advl_tech].sum(), storage_market_share.loc[2050,flow_tech].sum(), storage_market_share.loc[2050,salt_tech].sum(), storage_market_share.loc[2050,'Hydrogen FC'], storage_market_share.loc[2050,othr_tech].sum()]
   
   legend_elements = ['Flywheels','Compressed Air', 'Lithium-ion', 'Advanced Li', 'Flow \n Batteries', 'Molten \n Salt \n batteries', 'Hydrogen', 'Other']
   legend_elements2 = ['Flywheels','Compressed Air', 'Lithium-ion', '', 'Flow \n Batteries', 'Molten \n Salt \n batteries', '', 'Other']
   legend_elements3 = ['Flywheels','Compressed Air', 'Lithium-ion', 'Advanced Li', 'Flow Batteries', 'Molten Salt batteries', 'Hydrogen', 'Other']
   
   legend_elements.reverse()
   legend_elements2.reverse()
   legend_elements3.reverse()
   plotvar1.reverse()
   plotvar2.reverse()
   plotvar3.reverse()
   
   color_set = ("#a82c41", "#6a26d1", "#30942b", "#5c56d6", "#edce77", "#f0ba3c", "#babab8", "#4287f5")
   
   font = {'family' : 'sans-serif',
           'weight' : 'regular',
           'size'   : 7}
   
   plt.rc('font', **font)
   
   fig = plt.figure(figsize=(15, 5))
   gs = GridSpec(nrows=1, ncols=3)
   gs.update(wspace=0, hspace=0)
   fig.suptitle('Market shares of other (dedicated) electricity storage technologies', y=0.9, fontsize=14)
   
   ax0 = fig.add_subplot(gs[0, 0])
   ax0.set_title('2018', fontsize=14, y=0.45, fontweight='bold')
   ax0.text(-0.05, -1.15, 'a', fontsize=12)
   wedges, labels = ax0.pie(plotvar1, wedgeprops=dict(width=0.5), startangle=90, labels=legend_elements2, labeldistance=0.85, colors=color_set)
   for label in labels: label.set_horizontalalignment('center')
   
   ax1 = fig.add_subplot(gs[0, 1])
   ax1.set_title('2030', fontsize=14, y=0.45, fontweight='bold')
   ax1.text(-0.05, -1.15, 'b', fontsize=12)
   wedges, labels = ax1.pie(plotvar2, wedgeprops=dict(width=0.5), startangle=90, labels=legend_elements, labeldistance=0.85, colors=color_set)
   for label in labels: label.set_horizontalalignment('center')
   
   ax2 = fig.add_subplot(gs[0, 2])
   ax2.set_title('2050', fontsize=14, y=0.45, fontweight='bold')
   ax2.text(-0.05, -1.15, 'c', fontsize=12)
   wedges, labels = ax2.pie(plotvar3, wedgeprops=dict(width=0.5), startangle=90, labels=legend_elements, labeldistance=0.85, colors=color_set)
   for label in labels: label.set_horizontalalignment('center')
   
   plt.legend(legend_elements3, loc=8, bbox_to_anchor=(-.55, -0.15), ncol=4, frameon=False, fontsize=11)
   plt.savefig('output\\' + variant + '\\' + sa_settings + '\\graphs\\storshare.jpg', dpi=600)
