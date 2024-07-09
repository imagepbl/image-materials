import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from .constants import OUT_YEAR, VARIANT, OUTPUT_FOLDER, END_YEAR


idx = pd.IndexSlice

FONT = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 10}


COLOR_MATERIAL = {
    "Steel"     : "#16697a",
    "Aluminium" : "#79a3b1",
    "Cu"        : "#c7956d",
    "Plastics"  : "#ffd369",
    "Glass"     : "#bbbbbb",
    "Rubber"    : "#0a043c",
    "Co"        : "#314a73",
    "Li"        : "#d6d6d6",
    "Mn"        : "#6d5987",
    "Nd"        : "#e38846",
    "Ni"        : "#428a66",
    "Pb"        : "#5b6b5f",
    "Ti"        : "#daf2e0",
    "Wood"      : "#73523b"
    }

VEHICLE_LIST =       ['Bicycles', 'Cars',    'Planes',  'Trains',  'Buses',   'Rail Cargo', 'Air Cargo', 'Trucks',  'Ships']
COLORS_VEHICLES =    ["#343f5c", "#45537a",    "#576b9e",   "#8caaf5", "#d1deff", "#3d4a40",  "#516355", "#809c86", "#aed6b6"]
VEH_COLORS_DICT =    {VEHICLE_LIST[i]: COLORS_VEHICLES[i] for i in range(len(VEHICLE_LIST))} 
YEARS_SELECT =       list(range(2000,OUT_YEAR + 1))


def plot_transport_demand(passengerkms_Tpkms, tonkms_Mtkms, variant=VARIANT, output_folder=OUTPUT_FOLDER):
    plt.rc('font', **FONT)
    index = np.array(list(range(2000,OUT_YEAR + 1)))
    graph_fp = Path(output_folder, "graphs", "Pkm-Tkm.jpg")

    if variant == 'BL':
        ylim = [70,125]
    else:
        ylim = [70,125]

    plt.figure()
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,6), frameon=True)
    fig.suptitle('Passenger & Freight Tranport Demand in the SSP2 Baseline (in Pkm & Tkm)', y=1, fontsize=16)
    plt.subplots_adjust(wspace = 0.25, bottom = 0.15)

    ax1.set_title('Passenger Transport Demand (Tera Pkm/yr)', fontsize=12)
    ax1.set_ylim(0,ylim[0])
    ax1.set_ylabel('Tera person-kilometer', rotation='vertical', y=0.8, fontsize=11)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.plot(index, passengerkms_Tpkms['air'].unstack()[28].loc[2000:OUT_YEAR],    'blue',   label='Airplane',         linewidth=2)
    ax1.plot(index, passengerkms_Tpkms['biking'].unstack()[28].loc[2000:OUT_YEAR], 'green',  label='Bicycle',          linewidth=2)
    ax1.plot(index, passengerkms_Tpkms['bus'].unstack()[28].loc[2000:OUT_YEAR],    'gold',   label='Bus',              linewidth=2)
    ax1.plot(index, passengerkms_Tpkms['car'].unstack()[28].loc[2000:OUT_YEAR],    'red',    label='Car',              linewidth=2)
    ax1.plot(index, passengerkms_Tpkms['train'].unstack()[28].loc[2000:OUT_YEAR],  'purple', label='Train',            linewidth=2)
    ax1.plot(index, passengerkms_Tpkms['hst'].unstack()[28].loc[2000:OUT_YEAR],    'orange', label='High Speed Train', linewidth=2)
    ax1.legend(loc=2, bbox_to_anchor=(0.015, -0.05), ncol=3, frameon=False, fontsize=10)

    ax2.set_title('Freight Transport Demand (Tera Tkm/yr)', fontsize=12)
    ax2.set_ylim(0,ylim[1])
    ax2.set_ylabel('Tera ton-kilometer', rotation='vertical', y=0.8, fontsize=11)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.plot(index, tonkms_Mtkms['international shipping'].unstack()[28].loc[2000:OUT_YEAR] / 1000000,    'black',    label='Internl. Shipping', linewidth=2)
    ax2.plot(index, tonkms_Mtkms['inland shipping'].unstack()[28].loc[2000:OUT_YEAR]        / 1000000,    'brown',    label='Inland Shipping',   linewidth=2)
    ax2.plot(index, tonkms_Mtkms['freight train'].unstack()[28].loc[2000:OUT_YEAR]          / 1000000,    'violet',   label='Rail Cargo',        linewidth=2)
    ax2.plot(index, (tonkms_Mtkms['medium truck'].unstack()[28].loc[2000:OUT_YEAR] + tonkms_Mtkms['heavy truck'].unstack()[28].loc[2000:OUT_YEAR]) / 1000000, 'grey',     label='Truck',             linewidth=2)
    ax2.plot(index, tonkms_Mtkms['air cargo'].unstack()[28].loc[2000:OUT_YEAR]              / 1000000,    'darkcyan', label='Air Cargo',         linewidth=2)

    ax2.legend(loc=2, bbox_to_anchor=(0.015, -0.05), ncol=3, frameon=False, fontsize=10)
    plt.savefig(graph_fp, dpi=600, pad_inches=2)
    plt.show()



def plot_vehicle_material_composition(
    material_fractions_truck_MFT, material_fractions_truck_HFT, 
    material_fractions_bicycle, material_fractions_bus_midi,
    material_fractions_car, material_fractions_bus_reg, material_fractions_air_pas, material_fractions_rail_reg,
    material_fractions_rail_hst, material_fractions_air_frgt, material_fractions_rail_frgt, material_fractions_inland_ship,
    material_fractions_truck_LCV,
    output_folder=OUTPUT_FOLDER):


    year = 2018
    mul = 100
    material_fractions_truck = (material_fractions_truck_MFT + material_fractions_truck_HFT) / 2    # pre calculate the avreage of Heavy & Medium Trucks
    figure_fp = Path(output_folder, "graphs", "bars_compositon.png")

    plotvar_steel      = [material_fractions_car.loc[year, idx['ICE','Steel']]*mul,     material_fractions_bicycle.loc[year,'Steel']*mul,     material_fractions_bus_midi.loc[year, idx['ICE','Steel']]*mul,     material_fractions_bus_reg.loc[year, idx['ICE','Steel']]*mul,     material_fractions_air_pas.loc[year,'Steel']*mul,     material_fractions_rail_reg.loc[year,'Steel']*mul,     material_fractions_rail_hst.loc[year,'Steel']*mul,     0, material_fractions_air_frgt.loc[year,'Steel']*mul,     material_fractions_rail_frgt.loc[year,'Steel']*mul,     material_fractions_inland_ship.loc[year,'Steel']*mul,     material_fractions_truck_LCV.loc[year, idx['ICE','Steel']]*mul,     material_fractions_truck.loc[year, idx['ICE','Steel']]*mul]
    plotvar_aluminium  = [material_fractions_car.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_bicycle.loc[year,'Aluminium']*mul, material_fractions_bus_midi.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_bus_reg.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_air_pas.loc[year,'Aluminium']*mul, material_fractions_rail_reg.loc[year,'Aluminium']*mul, material_fractions_rail_hst.loc[year,'Aluminium']*mul, 0, material_fractions_air_frgt.loc[year,'Aluminium']*mul, material_fractions_rail_frgt.loc[year,'Aluminium']*mul, material_fractions_inland_ship.loc[year,'Aluminium']*mul, material_fractions_truck_LCV.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_truck.loc[year, idx['ICE','Aluminium']]*mul]
    plotvar_cu         = [material_fractions_car.loc[year, idx['ICE','Cu']]*mul,        material_fractions_bicycle.loc[year,'Cu']*mul,        material_fractions_bus_midi.loc[year, idx['ICE','Cu']]*mul,        material_fractions_bus_reg.loc[year, idx['ICE','Cu']]*mul,        material_fractions_air_pas.loc[year,'Cu']*mul,        material_fractions_rail_reg.loc[year,'Cu']*mul,        material_fractions_rail_hst.loc[year,'Cu']*mul,        0, material_fractions_air_frgt.loc[year,'Cu']*mul,        material_fractions_rail_frgt.loc[year,'Cu']*mul,        material_fractions_inland_ship.loc[year,'Cu']*mul,        material_fractions_truck_LCV.loc[year, idx['ICE','Cu']]*mul,        material_fractions_truck.loc[year, idx['ICE','Cu']]*mul]
    plotvar_plastics   = [material_fractions_car.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_bicycle.loc[year,'Plastics']*mul,  material_fractions_bus_midi.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_bus_reg.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_air_pas.loc[year,'Plastics']*mul,  material_fractions_rail_reg.loc[year,'Plastics']*mul,  material_fractions_rail_hst.loc[year,'Plastics']*mul,  0, material_fractions_air_frgt.loc[year,'Plastics']*mul,  material_fractions_rail_frgt.loc[year,'Plastics']*mul,  material_fractions_inland_ship.loc[year,'Plastics']*mul,  material_fractions_truck_LCV.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_truck.loc[year, idx['ICE','Plastics']]*mul]
    plotvar_glass      = [material_fractions_car.loc[year, idx['ICE','Glass']]*mul,     material_fractions_bicycle.loc[year,'Glass']*mul,     material_fractions_bus_midi.loc[year, idx['ICE','Glass']]*mul,     material_fractions_bus_reg.loc[year, idx['ICE','Glass']]*mul,     material_fractions_air_pas.loc[year,'Glass']*mul,     material_fractions_rail_reg.loc[year,'Glass']*mul,     material_fractions_rail_hst.loc[year,'Glass']*mul,     0, material_fractions_air_frgt.loc[year,'Glass']*mul,     material_fractions_rail_frgt.loc[year,'Glass']*mul,     material_fractions_inland_ship.loc[year,'Glass']*mul,     material_fractions_truck_LCV.loc[year, idx['ICE','Glass']]*mul,     material_fractions_truck.loc[year, idx['ICE','Glass']]*mul]
    plotvar_rubber     = [material_fractions_car.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_bicycle.loc[year,'Rubber']*mul,    material_fractions_bus_midi.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_bus_reg.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_air_pas.loc[year,'Rubber']*mul,    material_fractions_rail_reg.loc[year,'Rubber']*mul,    material_fractions_rail_hst.loc[year,'Rubber']*mul,    0, material_fractions_air_frgt.loc[year,'Rubber']*mul,    material_fractions_rail_frgt.loc[year,'Rubber']*mul,    material_fractions_inland_ship.loc[year,'Rubber']*mul,    material_fractions_truck_LCV.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_truck.loc[year, idx['ICE','Rubber']]*mul]

    barWidth = 0.5
    r1 = np.arange(13)

    plt.figure(figsize=(20, 6))
    fig, ax = plt.subplots(figsize=(20,6))
    plt.rc('font', **FONT)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(wspace = 0.1, bottom = 0.15, right = 0.80)
    ax.set_ylim(ymin=0, ymax=100)
    ax.bar(r1, plotvar_steel,     color=COLOR_MATERIAL["Steel"],     width=barWidth, edgecolor=None, label='2018')
    bottom = plotvar_steel
    ax.bar(r1, plotvar_aluminium, color=COLOR_MATERIAL["Aluminium"], width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    bottom = [a+b for a,b in zip(bottom,plotvar_aluminium)]
    ax.bar(r1, plotvar_cu,        color=COLOR_MATERIAL["Cu"],        width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    bottom = [a+b for a,b in zip(bottom,plotvar_cu)]
    ax.bar(r1, plotvar_plastics,  color=COLOR_MATERIAL["Plastics"],  width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    bottom = [a+b for a,b in zip(bottom,plotvar_plastics)]
    ax.bar(r1, plotvar_glass,     color=COLOR_MATERIAL["Glass"],     width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    bottom = [a+b for a,b in zip(bottom,plotvar_glass)]
    ax.bar(r1, plotvar_rubber,    color=COLOR_MATERIAL["Rubber"],    width=barWidth, bottom=bottom, edgecolor=None, label='2018')

    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth - 0.5 for r in range(13)], ['Cars', 'Bicycles', 'Midi Bus', 'Bus', 'Airplane', 'Rail', 'High Speed Train', '', 'Air Cargo', 'Rail Cargo', 'Ships', 'Light Truck', 'Other Trucks'], fontsize=11)
    plt.ylabel('vehicle composition (wt%)', fontsize=18, y=0.5)
    plt.title('Default material composition of different vehicles', y=1.08, fontsize=16)

    # Legend & text
    legend_elements = [matplotlib.patches.Patch(facecolor=COLOR_MATERIAL["Rubber"],    label='Rubber'),
                    matplotlib.patches.Patch(facecolor=COLOR_MATERIAL["Glass"],     label='Glass'),
                    matplotlib.patches.Patch(facecolor=COLOR_MATERIAL["Plastics"],  label='Plastics'),
                    matplotlib.patches.Patch(facecolor=COLOR_MATERIAL["Cu"],        label='Cu'),
                    matplotlib.patches.Patch(facecolor=COLOR_MATERIAL["Aluminium"], label='Aluminium'),
                    matplotlib.patches.Patch(facecolor=COLOR_MATERIAL["Steel"],     label='Steel')]
    plt.legend(handles=legend_elements, loc=2, bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=16)
    fig.savefig(figure_fp, dpi=600)
    plt.show()


def plot_battery(battery_shares_full, vehicle_materials, idx, OUT_YEAR, battery_weight_total_in,
                 output_folder=OUTPUT_FOLDER):
    # Plot panel (1 * 2) Basic stacked area chart: Global totals of all battery types & materials 

    color_battery = {
        "NiMH"                  : "#334fff",
        "LMO/LCO"               : "#058b37",
        "NMC"                   : "#f2cc3a",
        "NCA"                   : "#e89a18",
        "LFP"                   : "#db6400",
        "Li-S"                  : "#fd3a69",
        "Li-Ceramic"            : "#9d0191",
        "Li-Air"                : "#120078"
    }
    #prepare data first
    figure_fp = Path(OUTPUT_FOLDER, "graphs", "battery_market_panel_inflow_new.png")

    # pre-process & assign plot data
    graph1_data = battery_shares_full.T[YEARS_SELECT]
    graph1_data = graph1_data.rename(index={'Lithium-air':'Li-Air','Lithium Ceramic ':'Li-Ceramic','Lithium Sulfur':'Li-S'})

    graph2_data = vehicle_materials.loc[idx[:,:,'inflow',:,'battery',:,:],YEARS_SELECT].sum(axis=0, level=5)
    graph2_data_total = graph2_data.sum(axis=0)

    #loop to get fractional contribution
    for material in list(graph2_data.index):
        if graph2_data.loc[material].sum() > 0.01:
            graph2_data.loc[material] = graph2_data.loc[material] / graph2_data_total
        else: 
            graph2_data = graph2_data.drop(material)

    YEARS_SELECT         = list(range(1990,OUT_YEAR +1))
    window_size          = 10

    graph_data_original = battery_weight_total_in

    # duplicate data for the last available year (to make moving average window work)
    graph_data_extended = graph_data_original
    if (OUT_YEAR == END_YEAR):
        for year in range(1,window_size):
            year_used = OUT_YEAR + year
            add  = graph_data_original.loc[idx[:,OUT_YEAR],:]
            index = pd.MultiIndex.from_product([list(add.index.levels[0]),[year_used]])
            add.set_index(index,inplace=True)
            graph_data_extended = pd.concat([graph_data_extended, add])
                        
    graph3_data = graph_data_extended.sum(axis=1).unstack()[YEARS_SELECT]    # battery weight by vehicle (cars,bus,truck)
    graph4_data = graph_data_extended.sum(axis=0, level=1).T.rolling(window=window_size, axis=1).mean()[YEARS_SELECT]   # battery weight by drivetrain (HEV/PHEV/BEV)

    graph3_data.loc['Trucks',:] =  graph_data_extended.sum(axis=1).unstack()[YEARS_SELECT].loc[['HFT','MFT','LCV'],:].sum(axis=0).rolling(window=window_size).mean()
    graph3_data.loc['Buses',:]  =  graph_data_extended.sum(axis=1).unstack()[YEARS_SELECT].loc[['midi_bus','reg_bus'],:].sum(axis=0).rolling(window=window_size).mean()
    graph3_data.loc['Cars',:]   =  graph_data_extended.sum(axis=1).unstack()[YEARS_SELECT].loc[['car'],:].sum(axis=0).rolling(window=window_size).mean()

    graph3_data_total = graph3_data.loc[['Cars','Buses','Trucks']].sum(axis=0)
    graph4_data_total = graph4_data.sum(axis=0)


    graph3_data = graph3_data.div(graph3_data_total, axis=1)[YEARS_SELECT]
    graph4_data = graph4_data.div(graph4_data_total, axis=1)[YEARS_SELECT].drop(['ICE','FCV'])

    material_colorset = [COLOR_MATERIAL[i] for i in list(graph2_data.index)]
    battery_colorset  = [color_battery[i]  for i in list(graph1_data.index)]
    vehicle_colors_select = [VEH_COLORS_DICT['Trucks'],VEH_COLORS_DICT['Buses'],VEH_COLORS_DICT['Cars']]

    plt.figure()
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15,12), frameon=True)
    fig.suptitle('EV Battery Markets, Composition and Demand', y=1, fontsize=16)
    plt.subplots_adjust(wspace = 0.25, hspace = 0.25, bottom = 0.15)
    ax2nd = ax2.twinx()

    ax1.set_title('Market share of battery types (sales/inflow)', fontsize=12)
    ax1.set_ylim(0,100)
    ax1.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.stackplot(YEARS_SELECT, graph1_data[YEARS_SELECT] * 100, labels=list(graph1_data.index), colors=battery_colorset)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.20), loc='lower left', ncol=4, frameon=False, fontsize=10)

    ax2.set_title('Battery material composition & sales volume', fontsize=12)
    ax2.set_ylim(0,100)
    ax2.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
    ax2nd.set_ylabel('battery weight (kg)', rotation='vertical', x=0.1, y=0.78, fontsize=13)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2nd.margins(x=0)
    ax2.stackplot(YEARS_SELECT, graph2_data * 100, labels=list(graph2_data.index), colors=material_colorset)
    ax2nd.plot(YEARS_SELECT, battery_weight_total_in.sum(axis=1).sum(axis=0,level=1)[YEARS_SELECT], '--', color='black', linewidth=3, label="Sales (>)") 
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.20), loc='lower left', ncol=5, frameon=False, fontsize=10)
    ax2nd.legend(bbox_to_anchor=(1, -0.20), loc='lower right', ncol=1, frameon=False, fontsize=10)

    ax3.set_title('Market share (inflow) of batteries by vehicle type', fontsize=12)
    ax3.set_ylim(0,100)
    ax3.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.margins(x=0)
    ax3.stackplot(YEARS_SELECT, graph3_data.loc[['Trucks','Buses','Cars'],:] * 100, labels=['Trucks','Buses','Cars'], colors=vehicle_colors_select)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.18), loc='lower left', ncol=4, frameon=False, fontsize=10)

    ax4.set_title('Market share (inflow) of batteries by drivetrain type', fontsize=12)
    ax4.set_ylim(0,100)
    ax4.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
    ax4.yaxis.set_ticks_position('left')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.margins(x=0)
    ax4.stackplot(YEARS_SELECT, graph4_data * 100, labels=['BEV','HEV','PHEV','Trolley'], colors=['#ffce89','#fff76a','#adce74','#61b15a'])
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.18), loc='lower left', ncol=4, frameon=False, fontsize=10)

    plt.savefig(figure_fp, dpi=600)
    plt.show()



def plot_global_stocks_flows(vehicle_material_stock_global, vehicle_materials,
                             output_folder=OUTPUT_FOLDER):
    # PLot panel (2 * 2) Results 
    # 1) total vehicle weight (including batteries), Stock by vehicle
    # 2) total vehicle weight (including batteries), Stock by material
    # 3) Steel weight, Stock by hevicle + inflow/outflow
    # 4) Aluminium weight, Stock by vehicle + inflow/outflow

    #material = 'Steel'
    #test = vehicle_material_stock_global.loc[idx[material,['sea_shipping_small', 'sea_shipping_med','sea_shipping_large', 'sea_shipping_vl']],:].sum(axis=0)

    bulk_list =          ["Steel" ,  "Aluminium",   "Cu",   "Plastics",   "Glass",   "Rubber"]
    figure_fp = Path(output_folder, "graphs", "vehicle_material_results_panel.png")
             
    #grouping
    vehicle_material_stock_global = vehicle_materials.loc[idx[:,:,'stock',:,:,:,:],:].sum(axis=0, level=[5,6])
    for material in list(vehicle_material_stock_global.index.levels[0]):
        vehicle_material_stock_global.loc[(material,'Cars'),:]   = vehicle_material_stock_global.loc[idx[material,['ICE','HEV','PHEV','BEV','FCV']],:].sum(axis=0)
        vehicle_material_stock_global.loc[(material,'Trains'),:] = vehicle_material_stock_global.loc[idx[material,['rail_reg','rail_hst']],:].sum(axis=0)
        vehicle_material_stock_global.loc[(material,'Trucks'),:] = vehicle_material_stock_global.loc[idx[material,['LCV','MFT','HFT']],:].sum(axis=0)
        vehicle_material_stock_global.loc[(material,'Buses'),:]  = vehicle_material_stock_global.loc[idx[material,['reg_bus','midi_bus']],:].sum(axis=0)
        vehicle_material_stock_global.loc[(material,'Ships'),:]  = vehicle_material_stock_global.loc[idx[material,['inland_shipping', 'sea_shipping_small', 'sea_shipping_med','sea_shipping_large', 'sea_shipping_vl']],:].sum(axis=0)

    vehicle_material_stock_global = vehicle_material_stock_global.rename(index={"bicycle" : "Bicycles", "air_pas": "Planes", "rail_freight": "Rail Cargo", "air_freight" : "Air Cargo"})

    graph1_data = vehicle_material_stock_global.loc[idx[bulk_list,VEHICLE_LIST],YEARS_SELECT].sum(axis=0, level=1).reindex(VEHICLE_LIST)    # by vehicle

    # main text (estimate of share of other vehicles)
    car_share_in_total_stock = graph1_data.sum(axis=1)['Cars']/graph1_data.sum(axis=1).sum()

    graph2_data = vehicle_material_stock_global.loc[idx[bulk_list,VEHICLE_LIST],YEARS_SELECT].sum(axis=0, level=0)                          # by material
    graph3_data = vehicle_material_stock_global.loc[idx['Steel',VEHICLE_LIST],YEARS_SELECT].sum(axis=0, level=1).reindex(VEHICLE_LIST)      # Steel by vehicle
    graph4_data = vehicle_material_stock_global.loc[idx['Aluminium',VEHICLE_LIST],YEARS_SELECT].sum(axis=0, level=1).reindex(VEHICLE_LIST)  # Aluminium by vehicle

    graph3_data_in  = vehicle_materials.loc[idx[:,:,'inflow',:,:,'Steel',:],YEARS_SELECT].sum(axis=0, level=5).loc['Steel']
    graph3_data_out = vehicle_materials.loc[idx[:,:,'outflow',:,:,'Steel',:],YEARS_SELECT].sum(axis=0, level=5).loc['Steel']

    graph4_data_in  = vehicle_materials.loc[idx[:,:,'inflow',:,:,'Aluminium',:],YEARS_SELECT].sum(axis=0, level=5).loc['Aluminium']
    graph4_data_out = vehicle_materials.loc[idx[:,:,'outflow',:,:,'Aluminium',:],YEARS_SELECT].sum(axis=0, level=5).loc['Aluminium']

    graph3_data_in[[2006,2007,2008]].mean() # quick comparison to Cullen & Alwood 2012
    graph4_data_in[[2006,2007,2008]].mean() # quick comparison to Cullen & Alwood 2012

    material_colorset = [COLOR_MATERIAL[i] for i in list(graph2_data.index)]

    plt.figure()
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(17,15), frameon=True)
    fig.suptitle('Global Material Stocks & Flows in Vehicles', y=0.95, fontsize=16)
    plt.subplots_adjust(wspace = 0.22, bottom = 0.1)
    ax3nd = ax3.twinx()
    ax4nd = ax4.twinx()

    ax1.set_title('Total Vehicle Body Weight (Stock), by Vehicle', fontsize=12)
    ax1.set_ylabel('Total Vehicle Weight (kg)', rotation='vertical', y=0.75, fontsize=13)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.stackplot(YEARS_SELECT, graph1_data, labels=list(graph1_data.index), colors=COLORS_VEHICLES)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

    ax2.set_title('Total Vehicle Body Weight (Stock), by Material', fontsize=12)
    ax2.set_ylabel('Total Vehicle Weight (kg)', rotation='vertical', y=0.75, fontsize=13)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.stackplot(YEARS_SELECT, graph2_data, labels=list(graph2_data.index), colors=material_colorset)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

    ax3.set_title('Global Steel Stocks and Flows in Vehicles', fontsize=12)
    ax3.set_ylabel('Vehicle Steel Stock (kg)', rotation='vertical', y=0.78, fontsize=13)
    ax3nd.set_ylabel('Vehicle Steel Flows (kg/yr)', rotation='vertical', x=0.1, y=0.75, fontsize=13)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.margins(x=0)
    ax3nd.margins(x=0)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3nd.tick_params(axis='both', which='major', labelsize=12)
    ax3.stackplot(YEARS_SELECT, graph3_data, labels=list(graph3_data.index), colors=COLORS_VEHICLES)
    ax3nd.plot(YEARS_SELECT, graph3_data_in,  '--', color='black', linewidth=3, label="Total Inflow") 
    ax3nd.plot(YEARS_SELECT, graph3_data_out, '--', color='red',   linewidth=3, label="Total Outflow") 
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)
    ax3nd.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, frameon=False, fontsize=13)

    ax4.set_title('Global Aluminium Stocks and Flows in Vehicles', fontsize=12)
    ax4.set_ylabel('Vehicle Aluminium Stock (kg)', rotation='vertical', y=0.75, fontsize=13)
    ax4nd.set_ylabel('Vehicle Aluminium Flows (kg/yr)', rotation='vertical', x=0.1, y=0.72, fontsize=13)
    ax4.yaxis.set_ticks_position('left')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.margins(x=0)
    ax4nd.margins(x=0)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4nd.tick_params(axis='both', which='major', labelsize=12)
    ax4.stackplot(YEARS_SELECT, graph4_data, labels=list(graph4_data.index), colors=COLORS_VEHICLES)
    ax4nd.plot(YEARS_SELECT, graph4_data_in,  '--', color='black', linewidth=3, label="Total Inflow") 
    ax4nd.plot(YEARS_SELECT, graph4_data_out, '--', color='red',   linewidth=3, label="Total Outflow") 
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)
    ax4nd.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, frameon=False, fontsize=13)

    plt.show()
    plt.savefig(figure_fp, dpi=600)


def plot_net_steel_additions_stocks(vehicle_materials, output_folder=OUTPUT_FOLDER):
    # Plot panel (1 * 2) 3 regional Stock development & consequences i.t.o. inflow/outflow
    import matplotlib.ticker as ticker
    years_average_now   = [2016,2017,2018,2019,2020] 
    years_average_end   = [OUT_YEAR -4,OUT_YEAR -3,OUT_YEAR -2,OUT_YEAR -1,OUT_YEAR] 

    material_select = 'Steel'
    vehcile_list       = ['bicycle', 'rail_reg', 'rail_hst', 'reg_bus', 'midi_bus']           #only those vehicles which go negative by the end of the scenario (may add: 'sea_shipping_small', 'sea_shipping_med', 'sea_shipping_large', 'sea_shipping_vl')
    regions_developed  = [1,2,3,11,12,13,14,15,16,17,24]
    regions_developing = [4,5,6,7,8,9,10,18,19,21,22,25,26]
    regions_chin_jap   = [20,23]
                        
    #apply vehcile grouping first (same as panel above)
    material_flows_grouped = vehicle_materials.loc[idx[:,:,['inflow','outflow'],:,:, material_select,vehcile_list],:].sum(axis=0, level=[2,3,6]) # vehicle_materials in kg
    for flow in list(material_flows_grouped.index.levels[0]):
        for region in list(material_flows_grouped.index.levels[1]):
            #material_flows_grouped.loc[(flow, region,'Cars'),:]   = material_flows_grouped.loc[idx[flow, region,['ICE','HEV','PHEV','BEV','FCV']],:].sum(axis=0)
            material_flows_grouped.loc[(flow, region,'Trains'),:] = material_flows_grouped.loc[idx[flow, region,['rail_reg','rail_hst']],:].sum(axis=0)
            #material_flows_grouped.loc[(flow, region,'Trucks'),:] = material_flows_grouped.loc[idx[flow, region,['LCV','MFT','HFT']],:].sum(axis=0)
            material_flows_grouped.loc[(flow, region,'Buses'),:]  = material_flows_grouped.loc[idx[flow, region,['reg_bus','midi_bus']],:].sum(axis=0)
            #material_flows_grouped.loc[(flow, region,'Ships'),:]  = material_flows_grouped.loc[idx[flow, region,['inland_shipping', 'sea_shipping_small', 'sea_shipping_med','sea_shipping_large', 'sea_shipping_vl']],:].sum(axis=0)

    material_flows_grouped = material_flows_grouped.rename(index={"bicycle" : "Bicycles", "air_pas": "Planes", "rail_freight": "Rail Cargo", "air_freight" : "Air Cargo"})

    # pre-calculate & assign plot data
    in_steel_developed_now   = material_flows_grouped.loc[idx['inflow', regions_developed,  VEHICLE_LIST], years_average_now].sum(axis=0, level=2).mean(axis=1)
    in_steel_developing_now  = material_flows_grouped.loc[idx['inflow', regions_developing, VEHICLE_LIST], years_average_now].sum(axis=0, level=2).mean(axis=1)
    in_steel_chin_jap_now    = material_flows_grouped.loc[idx['inflow', regions_chin_jap,   VEHICLE_LIST], years_average_now].sum(axis=0, level=2).mean(axis=1)

    out_steel_developed_now  = material_flows_grouped.loc[idx['outflow', regions_developed,  VEHICLE_LIST], years_average_now].sum(axis=0, level=2).mean(axis=1)
    out_steel_developing_now = material_flows_grouped.loc[idx['outflow', regions_developing, VEHICLE_LIST], years_average_now].sum(axis=0, level=2).mean(axis=1)
    out_steel_chin_jap_now   = material_flows_grouped.loc[idx['outflow', regions_chin_jap,   VEHICLE_LIST], years_average_now].sum(axis=0, level=2).mean(axis=1)

    in_steel_developed_end   = material_flows_grouped.loc[idx['inflow', regions_developed,  VEHICLE_LIST], years_average_end].sum(axis=0, level=2).mean(axis=1)
    in_steel_developing_end  = material_flows_grouped.loc[idx['inflow', regions_developing, VEHICLE_LIST], years_average_end].sum(axis=0, level=2).mean(axis=1)
    in_steel_chin_jap_end    = material_flows_grouped.loc[idx['inflow', regions_chin_jap,   VEHICLE_LIST], years_average_end].sum(axis=0, level=2).mean(axis=1)

    out_steel_developed_end  = material_flows_grouped.loc[idx['outflow', regions_developed,  VEHICLE_LIST], years_average_end].sum(axis=0, level=2).mean(axis=1)
    out_steel_developing_end = material_flows_grouped.loc[idx['outflow', regions_developing, VEHICLE_LIST], years_average_end].sum(axis=0, level=2).mean(axis=1)
    out_steel_chin_jap_end   = material_flows_grouped.loc[idx['outflow', regions_chin_jap,   VEHICLE_LIST], years_average_end].sum(axis=0, level=2).mean(axis=1)

    diff_steel_developed_now  = in_steel_developed_now - out_steel_developed_now
    diff_steel_developing_now = in_steel_developing_now - out_steel_developing_now
    diff_steel_chin_jap_now   = in_steel_chin_jap_now - out_steel_chin_jap_now

    diff_steel_developed_end  = in_steel_developed_end - out_steel_developed_end
    diff_steel_developing_end = in_steel_developing_end - out_steel_developing_end
    diff_steel_chin_jap_end   = in_steel_chin_jap_end - out_steel_chin_jap_end


    plotvar_bus = [diff_steel_developed_now['Buses'],diff_steel_developing_now['Buses'], diff_steel_chin_jap_now['Buses'],                0, diff_steel_developed_end['Buses'], diff_steel_developing_end['Buses'], diff_steel_chin_jap_end['Buses']]
    plotvar_cyc = [diff_steel_developed_now['Bicycles'],diff_steel_developing_now['Bicycles'], diff_steel_chin_jap_now['Bicycles'],       0, diff_steel_developed_end['Bicycles'], diff_steel_developing_end['Bicycles'], diff_steel_chin_jap_end['Bicycles']]
    #plotvar_car = [diff_steel_developed_now['Cars'],diff_steel_developing_now['Cars'], diff_steel_chin_jap_now['Cars'],                   0, diff_steel_developed_end['Cars'], diff_steel_developing_end['Cars'], diff_steel_chin_jap_end['Cars']]
    #plotvar_trk = [diff_steel_developed_now['Trucks'],diff_steel_developing_now['Trucks'], diff_steel_chin_jap_now['Trucks'],             0, diff_steel_developed_end['Trucks'], diff_steel_developing_end['Trucks'], diff_steel_chin_jap_end['Trucks']]
    plotvar_trn = [diff_steel_developed_now['Trains'],diff_steel_developing_now['Trains'], diff_steel_chin_jap_now['Trains'],             0, diff_steel_developed_end['Trains'], diff_steel_developing_end['Trains'], diff_steel_chin_jap_end['Trains']]
    #plotvar_rlc = [diff_steel_developed_now['Rail Cargo'],diff_steel_developing_now['Rail Cargo'], diff_steel_chin_jap_now['Rail Cargo'], 0, diff_steel_developed_end['Rail Cargo'], diff_steel_developing_end['Rail Cargo'], diff_steel_chin_jap_end['Rail Cargo']]
    #plotvar_shp = [diff_steel_developed_now['Ships'],diff_steel_developing_now['Ships'], diff_steel_chin_jap_now['Ships'],                0, diff_steel_developed_end['Ships'], diff_steel_developing_end['Ships'], diff_steel_chin_jap_end['Ships']]

    #split the data into positive & negative bars
    def pos_neg(data):
        data_pos = []
        data_neg = []
        for item in range(0,len(data)):
            if data[item] > 0:
                data_pos.append(data[item])
                data_neg.append(0)
            else:
                data_neg.append(data[item])
                data_pos.append(0)
        return data_pos, data_neg

    plotvar_bus_pos, plotvar_bus_neg = pos_neg(plotvar_bus)
    plotvar_cyc_pos, plotvar_cyc_neg = pos_neg(plotvar_cyc)
    #plotvar_car_pos, plotvar_car_neg = pos_neg(plotvar_car)
    #plotvar_trk_pos, plotvar_trk_neg = pos_neg(plotvar_trk)
    plotvar_trn_pos, plotvar_trn_neg = pos_neg(plotvar_trn)
    #plotvar_rlc_pos, plotvar_rlc_neg = pos_neg(plotvar_rlc)
    #plotvar_shp_pos, plotvar_shp_neg = pos_neg(plotvar_shp)

    barWidth = 0.5
    r1 = np.arange(7)

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rc('font', **FONT)
    plt.yticks(fontsize=10)
    plt.subplots_adjust(wspace = 0.1, bottom = 0.15, right = 0.80)
    ax.set_ylim(ymin=-300000002, ymax=1400000002)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100000000))
    ax.bar(r1, plotvar_bus_pos, color=VEH_COLORS_DICT["Buses"],    width=barWidth, edgecolor=None, label='2018')
    ax.bar(r1, plotvar_bus_neg, color=VEH_COLORS_DICT["Buses"],    width=barWidth, edgecolor=None, label='2018')
    bottom = plotvar_bus_pos
    floor = plotvar_bus_neg
    ax.bar(r1, plotvar_cyc_pos, color=VEH_COLORS_DICT["Bicycles"], width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    ax.bar(r1, plotvar_cyc_neg, color=VEH_COLORS_DICT["Bicycles"], width=barWidth, bottom=floor,  edgecolor=None, label='2018')
    bottom = [a+b for a,b in zip(bottom,plotvar_cyc_pos)]
    floor =  [a+b for a,b in zip(floor, plotvar_cyc_neg)]
    #ax.bar(r1, plotvar_car_pos, color=VEH_COLORS_DICT["Cars"],     width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    #ax.bar(r1, plotvar_car_neg, color=VEH_COLORS_DICT["Cars"],     width=barWidth, bottom=floor,  edgecolor=None, label='2018')
    #bottom = [a+b for a,b in zip(bottom,plotvar_car_pos)]
    #floor =  [a+b for a,b in zip(floor, plotvar_car_neg)]
    ax.bar(r1, plotvar_trn_pos,  color=VEH_COLORS_DICT["Trains"],  width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    ax.bar(r1, plotvar_trn_neg,  color=VEH_COLORS_DICT["Trains"],  width=barWidth, bottom=floor, edgecolor=None, label='2018')
    bottom = [a+b for a,b in zip(bottom,plotvar_trn_pos)]
    floor =  [a+b for a,b in zip(floor, plotvar_trn_neg)]
    #ax.bar(r1, plotvar_trk_pos, color=VEH_COLORS_DICT["Trucks"],   width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    #ax.bar(r1, plotvar_trk_neg, color=VEH_COLORS_DICT["Trucks"],   width=barWidth, bottom=floor, edgecolor=None, label='2018')
    #bottom = [a+b for a,b in zip(bottom,plotvar_trk_pos)]
    #floor  = [a+b for a,b in zip(floor, plotvar_trk_neg)]
    #ax.bar(r1, plotvar_rlc_pos, color=VEH_COLORS_DICT["Rail Cargo"], width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    #ax.bar(r1, plotvar_rlc_neg, color=VEH_COLORS_DICT["Rail Cargo"], width=barWidth, bottom=floor,  edgecolor=None, label='2018')
    #bottom = [a+b for a,b in zip(bottom,plotvar_rlc_pos)]
    #floor =  [a+b for a,b in zip(floor, plotvar_rlc_neg)]
    #ax.bar(r1, plotvar_shp_pos,    color=VEH_COLORS_DICT["Ships"],   width=barWidth, bottom=bottom, edgecolor=None, label='2018')
    #ax.bar(r1, plotvar_shp_neg,    color=VEH_COLORS_DICT["Ships"],   width=barWidth, bottom=floor, edgecolor=None, label='2018')
    plt.plot([-0.5, 6.5], [0, 0], 'k-', lw=0.5)

    #Added text
    ax.text(1.1, -500000000, "Current (2020)", ha="center", va="center", rotation=0, size=10)
    ax.text(5, -500000000, str(OUT_YEAR),           ha="center", va="center", rotation=0, size=10)
    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="g", lw=0.5)
    ax.text(3, 55000000,  "Sink",   ha="center", va="center", rotation=90,  size=6, bbox=bbox_props)
    ax.text(3, -95000000, "Source", ha="center", va="center", rotation=-90, size=6, bbox=bbox_props)

    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth - 0.5 for r in range(7)], ['Steady', 'Developing', 'China+Japan', '', 'Steady', 'Developing', 'China+Japan'], fontsize=8)
    plt.ylabel('Net additions to vehicle stocks (kg steel)', fontsize=11, y=0.5)
    plt.title('Net additions of steel to selected in-use vehicle stocks', y=1.08, fontsize=14)


    # Legend & text
    legend_elements = [#matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Ships"],       label='Ships'),
                    #matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Rail Cargo"],  label='Rail Cargo'),
                    #matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Trucks"],      label='Trucks'),
                    matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Trains"],      label='Trains'),                
                    #matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Cars"],        label='Cars'),
                    matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Bicycles"],    label='Bicycles'),
                    matplotlib.patches.Patch(facecolor=VEH_COLORS_DICT["Buses"],       label='Buses')]
    plt.legend(handles=legend_elements, loc=2, bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=13)
    fig.savefig(OUTPUT_FOLDER + '\\graphs\\bars_source_sink_steel.png', dpi=600)
    plt.show()


def plot_stock_weight_intensity(car_total_nr, region_list,
                                vehicleshare_cars, vehicle_weight_kg_car, trucks_MFT_vshares, vehicle_weight_kg_LCV,
                                trucks_LCV_nr, trucks_HFT_nr, trucks_HFT_vshares, vehicle_weight_kg_HFT, vehicle_weight_kg_MFT, trucks_MFT_nr,
                                bus_regl_nr, buses_regl_vshares, vehicle_weight_kg_bus, vehicle_weight_kg_midi, bus_midi_nr, buses_midi_vshares,
                                bikes_nr, air_pas_nr, rail_reg_nr, rail_hst_nr, air_freight_nr, rail_freight_nr, inland_ship_nr,
                                vehicle_weight_kg_bicycle, vehicle_weight_kg_air_pas, vehicle_weight_kg_rail_reg, vehicle_weight_kg_rail_hst,
                                vehicle_weight_kg_air_frgt, vehicle_weight_kg_rail_frgt, vehicle_weight_kg_inland_ship,
                                ship_small_nr, car_pkms, trucks_LCV_tkm, trucks_MFT_tkm, bus_regl_pkms, bus_midi_pkms, passengerkms_Tpkms,
                                air_freight_tkms, tonkms_Mtkms, trucks_HFT_tkm, weight_boats, ship_medium_nr, ship_large_nr, ship_vlarge_nr,
                                output_folder=OUTPUT_FOLDER
    ):
    # Plot on the materials in-use  by vecihle, per tkm or pkm provided (each year)
    period = [2015, 2016, 2017, 2018, 2019, 2020]
    figure_fp = Path(output_folder, "graphs", "bars_vehicle_material_intensity.png")

    #pre-calculate total weight of vehicles (accounting for different vehcile sub-types)
    cars_weight = 0
    rbus_weight = 0
    mibs_weight = 0
    lcvt_weight = 0
    trck_weight = 0

    for veh_type in ['ICE','HEV','PHEV','BEV','FCV']:
        cars_weight = cars_weight + car_total_nr[region_list].mul(vehicleshare_cars[veh_type].unstack()).loc[period,:].sum(axis=1).mean()   * vehicle_weight_kg_car[veh_type].loc[period].mean() # MIND for the time being, the non-weighted average weight is used here
        lcvt_weight = lcvt_weight + trucks_LCV_nr[region_list].mul(trucks_MFT_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_LCV[veh_type].loc[period].mean() # LCV uses MFT vshares
        trck_weight = trck_weight + trucks_HFT_nr[region_list].mul(trucks_HFT_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_HFT[veh_type].loc[period].mean()
        trck_weight = trck_weight + trucks_MFT_nr[region_list].mul(trucks_MFT_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_MFT[veh_type].loc[period].mean() # MFT is summed with HFT

    for veh_type in ['ICE','HEV','PHEV','BEV','FCV']:
        rbus_weight = rbus_weight + bus_regl_nr[region_list].mul(buses_regl_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_bus[veh_type].loc[period].mean() 
        mibs_weight = mibs_weight + bus_midi_nr[region_list].mul(buses_midi_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_midi[veh_type].loc[period].mean() 

    bike_weight = bikes_nr.loc[period, region_list].sum(axis=1).mean()        * vehicle_weight_kg_bicycle.loc[period, "bicycle"].mean()
    airp_weight = air_pas_nr.loc[period, region_list].sum(axis=1).mean()      * vehicle_weight_kg_air_pas.loc[period, "air_pas"].mean()
    trai_weight = rail_reg_nr.loc[period, region_list].sum(axis=1).mean()     * vehicle_weight_kg_rail_reg.loc[period, "rail_reg"].mean()
    hstr_weight = rail_hst_nr.loc[period, region_list].sum(axis=1).mean()     * vehicle_weight_kg_rail_hst.loc[period, "rail_hst"].mean()
    airc_weight = air_freight_nr.loc[period, region_list].sum(axis=1).mean()  * vehicle_weight_kg_air_frgt.loc[period, "air_freight"].mean()
    rlfr_weight = rail_freight_nr.loc[period, region_list].sum(axis=1).mean() * vehicle_weight_kg_rail_frgt.loc[period, "rail_freight"].mean()
    ship_weight = (inland_ship_nr.loc[period, region_list].sum(axis=1).mean() * vehicle_weight_kg_inland_ship.loc[period, "inland_shipping"].mean()) + (ship_small_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Small'].mean()) +   (ship_medium_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Medium'].mean())  +  (ship_large_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Large'].mean())     + (ship_vlarge_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Very Large'].mean())         

    matint_cars = cars_weight / (car_pkms.loc[period,:].sum(axis=1).mean() * 1000000000000)
    matint_lcvt = lcvt_weight / (trucks_LCV_tkm.loc[period, region_list].sum(axis=1).mean() * 1000000)
    matint_trck = trck_weight / (trucks_MFT_tkm.loc[period, region_list].sum(axis=1).mean() * 1000000 + trucks_HFT_tkm.loc[period, region_list].sum(axis=1).mean() * 1000000)
    matint_rbus = rbus_weight / (bus_regl_pkms.loc[period,:].sum(axis=1).mean() * 1000000000000)
    matint_mibs = mibs_weight / (bus_midi_pkms.loc[period,:].sum(axis=1).mean() * 1000000000000)
    matint_bike = bike_weight / (passengerkms_Tpkms['biking'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
    matint_airp = airp_weight / (passengerkms_Tpkms['air'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
    matint_trai = trai_weight / (passengerkms_Tpkms['train'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
    matint_hstr = hstr_weight / (passengerkms_Tpkms['hst'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
    matint_airc = airc_weight / (air_freight_tkms.loc[period, region_list].sum(axis=1).mean() * 1000000)
    matint_rlfr = rlfr_weight / (tonkms_Mtkms['freight train'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000)
    matint_ship = ship_weight / (tonkms_Mtkms['international shipping'].unstack().loc[period,region_list].sum(axis=1).mean() * 1000000 + tonkms_Mtkms['inland shipping'].unstack().loc[period,region_list].sum(axis=1).mean() * 1000000)

    mul = 1000
    plotvar_matint  = [matint_cars * mul,   matint_bike * mul, matint_mibs * mul, matint_rbus * mul, matint_airp * mul, matint_trai * mul, matint_hstr * mul, 0, matint_airc * mul, matint_rlfr * mul, matint_ship * mul, matint_lcvt * mul, matint_trck * mul]

    barWidth = 0.5
    r1 = np.arange(13)

    plt.figure(figsize=(20, 6))
    fig, ax = plt.subplots(figsize=(20,6))
    plt.rc('font', **FONT)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(wspace = 0.1, bottom = 0.15, right = 0.80)
    ax.set_ylim(ymin=0, ymax=100)
    ax.bar(r1, plotvar_matint,     color=COLOR_MATERIAL["Steel"],     width=barWidth, edgecolor=None, label='2018')
    bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="b", lw=0.5)
    ax.text(11, 100,  "180",   ha="center", va="center", rotation=90,  size=10, bbox=bbox_props)

    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth - 0.5 for r in range(13)], ['Cars', 'Bicycles', 'Midi Bus', 'Bus', 'Airplane', 'Rail', 'High Speed Train', '', 'Air Cargo', 'Rail Cargo', 'Ships', 'Light Truck', 'Other Trucks'], fontsize=11)
    plt.ylabel('Vehicle weight intensity (gram/pkm or gram/tkm)', fontsize=14, x=-0.1, y=0.5)
    plt.title('Stock weight intensity of different vehicles (in-use stock weight per unit of annual transport demand)', y=1.08, fontsize=14)

    fig.savefig(figure_fp, dpi=600)
    plt.show()


def plot_battery_stocks_flows(battery_weight_regional_stock, regions_developed, regions_developing, regions_chin_jap,
                              battery_weight_regional_in, battery_weight_regional_out,
                              output_folder=OUTPUT_FOLDER):
    # Figures on results
    # Plot panel (1 * 2) Basic stacked area chart: REgional stock of battery (by weight) and the corresponding regional in/outflow 
    region_colorset = ['#682c0e','#c24914','#fc8621']
    YEARS_SELECT_narrow  = list(range(2000,OUT_YEAR + 1))
    YEARS_SELECT_broad   = list(range(1995,OUT_YEAR + 6))
    figure_fp = Path(output_folder, "graphs", "battery_market_panel_regional_stock_results.jpg")

    # pre-process & assign plot data
    graph1_data     = pd.DataFrame(index=YEARS_SELECT, columns=['Steady','Developing','China+Japan'])
    graph2_data_in  = pd.DataFrame(index=YEARS_SELECT, columns=['Steady','Developing','China+Japan'])
    graph2_data_out = pd.DataFrame(index=YEARS_SELECT, columns=['Steady','Developing','China+Japan'])

    graph1_data['Steady']         = battery_weight_regional_stock.loc[YEARS_SELECT_narrow, regions_developed].sum(axis=1)
    graph1_data['Developing']     = battery_weight_regional_stock.loc[YEARS_SELECT_narrow, regions_developing].sum(axis=1)
    graph1_data['China+Japan']    = battery_weight_regional_stock.loc[YEARS_SELECT_narrow, regions_chin_jap].sum(axis=1)

    graph2_data_in['Steady']      = battery_weight_regional_in.loc[YEARS_SELECT_narrow, regions_developed].sum(axis=1).rolling(window=5).mean().loc[YEARS_SELECT_narrow]
    graph2_data_in['Developing']  = battery_weight_regional_in.loc[YEARS_SELECT_narrow, regions_developing].sum(axis=1).rolling(window=5).mean().loc[YEARS_SELECT_narrow]
    graph2_data_in['China+Japan'] = battery_weight_regional_in.loc[YEARS_SELECT_narrow, regions_chin_jap].sum(axis=1).rolling(window=5).mean().loc[YEARS_SELECT_narrow]

    graph2_data_out['Steady']      = battery_weight_regional_out.loc[YEARS_SELECT_narrow, regions_developed].sum(axis=1).rolling(window=5).mean().loc[YEARS_SELECT_narrow]
    graph2_data_out['Developing']  = battery_weight_regional_out.loc[YEARS_SELECT_narrow, regions_developing].sum(axis=1).rolling(window=5).mean().loc[YEARS_SELECT_narrow]
    graph2_data_out['China+Japan'] = battery_weight_regional_out.loc[YEARS_SELECT_narrow, regions_chin_jap].sum(axis=1).rolling(window=5).mean().loc[YEARS_SELECT_narrow]


    plt.figure()
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,6), frameon=True)
    fig.suptitle('Regional EV Battery Stock & Flows', y=1, fontsize=16)
    plt.subplots_adjust(wspace = 0.25, bottom = 0.15)

    ax1.set_title('Battery in-use stocks by regional group', fontsize=12)
    ax1.set_ylabel('weight of battery stocks (kg)', rotation='vertical', y=0.65, fontsize=13)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.margins(x=0)
    ax1.stackplot(YEARS_SELECT, graph1_data.T, labels=list(graph1_data.columns), colors=region_colorset)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.15), loc='lower left', ncol=4, frameon=False, fontsize=10)

    ax2.set_title('Battery inflow/outflow by regional group', fontsize=12)
    ax2.set_ylabel('weight of battery flows (kg/yr)', rotation='vertical', y=0.65, fontsize=13)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.margins(x=0)
    ax2.plot(YEARS_SELECT, graph2_data_in['Steady'],       label='Steady (inflow)',    color=region_colorset[0])
    ax2.plot(YEARS_SELECT, graph2_data_out['Steady'],      '--', label='Steady (outflow)',    color=region_colorset[0])
    ax2.plot(YEARS_SELECT, graph2_data_in['Developing'],   label='Developing (inflow)', color=region_colorset[1])
    ax2.plot(YEARS_SELECT, graph2_data_out['Developing'],  '--', label='Developing (outflow)', color=region_colorset[1])
    ax2.plot(YEARS_SELECT, graph2_data_in['China+Japan'],  label='China+Japan (inflow)', color=region_colorset[2])
    ax2.plot(YEARS_SELECT, graph2_data_out['China+Japan'], '--', label='China+Japan (outflow)', color=region_colorset[2])
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(-0.05, -0.20), loc='lower left', ncol=3, frameon=False, fontsize=9)

    plt.savefig(figure_fp, dpi=600)
    plt.show()
