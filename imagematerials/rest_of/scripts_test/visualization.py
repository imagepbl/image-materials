import matplotlib.pyplot as plt

# plot for if all image regions are available and not grouped togehter
def plot_overview_figure_total_consumption_region(model, region: str):

    fig, ax = plt.subplots()             
    ax.plot(model.projection_per_region_total[region] + model.image_mat_data[region].loc[2012:], 
            linestyle = '--', color = 'blue', label = 'projected total consumption')
    ax.plot(model.historic_consumption_data[region],
            linestyle = '-', color = 'black', label = 'historic consumption')  
    ax.plot(model.image_mat_data[region].loc[1971:],
            linestyle = '--', color = 'red', label = 'buildings, vehicles, electricity (MAT)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total consumption')
    ax.legend()
    plt.show()

    
