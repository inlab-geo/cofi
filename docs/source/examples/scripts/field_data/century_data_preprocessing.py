"""
Pre-processing Century Deposit Data
===================================

Note: The preprocessing code is adapted from a SimPEG example authored
by Lindsey Heagy and presented at Transform 2020. `Original
Materials <https://curvenote.com/@simpeg/transform-2020-simpeg-tutorial/!6DDumb03Le6D8N8xuJNs>`__

This notebook cpatures the preprocessing of the dataset being used in
`Century Data DCIP example <pygimli_century_dcip.ipynb>`__. More
specifically, the DC and IP data on line ``46800E`` are organized and
stored into file ``century_dcip_data.txt`` with the following
attributes: - a_location - b_location - m_location - n_location -
apparent resistivity - apparent resistivity standard deviation -
apparent chargeability - apparent chargeability standard deviation -
geometric_factor

"""


######################################################################
# 1. Load the dataset
# -------------------
# 
# 1.1. Download dataset and import modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# download the dataset

# !git clone https://github.com/simpeg/transform-2020-simpeg.git
# %cd transform-2020-simpeg/

######################################################################
#

import os
import shutil
import numpy as np

######################################################################
#


######################################################################
# 1.2. See what’s in the data folder
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# overview of dataset files/folders
os.listdir('century')

######################################################################
#

# data files for the chosen line
line = "46800E"
os.listdir(os.path.join('century',line))

######################################################################
#

# files we are going to load
dc_data_file = f"./century/{line}/{line[:-1]}POT.OBS"
ip_data_file = f"./century/{line}/{line[:-1]}IP.OBS"

dc_data_file, ip_data_file

######################################################################
#

# copy geologic section data
shutil.copy("century/geologic_section.csv", "../century_geologic_section.csv")

######################################################################
#

# copy reference images
shutil.copy("images/Mutton-Figure1-1.png", "../Mutton-Figure1-1.png")
shutil.copy("images/Mutton-Figure2-1.png", "../Mutton-Figure2-1.png")

######################################################################
#


######################################################################
# 1.3. Define utility loader function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# Utility load function - to be used for both DC and IP data files

def read_dcip_data(filename, verbose=True):
    """
    Read in a .OBS file from the Century data set into a python dictionary. 
    The format is the old UBC-GIF DCIP format.
    
    Parameters
    ----------
    filename : str
        Path to the file to be parsed
    
    verbose: bool
        Print some things? 
    
    
    Returns
    -------
    dict
        A dictionary with the locations of
        - a_locations: the positive source electrode locations (numpy array) 
        - b_locations: the negative source electrode locations (numpy array) 
        - m_locations: the receiver locations (list of numpy arrays)
        - n_locations: the receiver locations (list of numpy arrays)
        - observed_data: observed data (list of numpy arrays)
        - standard_deviations: assigned standard deviations (list of numpy arrays)
        - n_sources: number of sources (int)
    
    """
    
    # read in the text file as a numpy array of strings (each row is an entry)
    contents = np.genfromtxt(filename, delimiter=' \n', dtype=str)
    
    # the second line has the number of sources, current, and data type (voltages if 1)
    n_sources = int(contents[1].split()[0])
    
    if verbose is True: 
        print(f"number of sources: {n_sources}")
    
    # initialize storage for the electrode locations and data
    a_locations = np.zeros(n_sources)
    b_locations = np.zeros(n_sources)
    m_locations = []
    n_locations = []
    observed_data = []
    standard_deviations = []
    
    # index to track where we have read in content 
    content_index = 1 
    
    # loop over sources 
    for i in range(n_sources):
        # start by reading in the source info 
        content_index = content_index + 1  # read the next line
        a_location, b_location, nrx = contents[content_index].split()  # this is a string
        
        # convert the strings to a float for locations and an int for the number of receivers
        a_locations[i] = float(a_location)
        b_locations[i] = float(b_location)
        nrx = int(nrx)

        if verbose is True: 
            print(f"Source {i}: A-loc: {a_location}, B-loc: {b_location}, N receivers: {nrx}")

        # initialize space for receiver locations, observed data associated with this source
        m_locations_i, n_locations_i = np.zeros(nrx), np.zeros(nrx)
        observed_data_i, standard_deviations_i = np.zeros(nrx), np.zeros(nrx)

        # read in the receiver info 
        for j in range(nrx):
            content_index = content_index + 1  # read the next line
            m_location, n_location, datum, std = contents[content_index].split()

            # convert the locations and data to floats, and store them
            m_locations_i[j] = float(m_location)
            n_locations_i[j] = float(n_location)
            observed_data_i[j] = float(datum)
            standard_deviations_i[j] = float(std)

        # append the receiver info to the lists
        m_locations.append(m_locations_i)
        n_locations.append(n_locations_i)
        observed_data.append(observed_data_i)
        standard_deviations.append(standard_deviations_i)
    
    return {
        "a_locations": a_locations,
        "b_locations": b_locations, 
        "m_locations": m_locations,
        "n_locations": n_locations,
        "observed_data": observed_data, 
        "standard_deviations": standard_deviations,
        "n_sources": n_sources, 
    }

######################################################################
#


######################################################################
# 2. Apparent Resistivity
# -----------------------
# 
# 2.1. Load voltage data
# ~~~~~~~~~~~~~~~~~~~~~~
# 

dc_data_dict = read_dcip_data(dc_data_file)

######################################################################
#


######################################################################
# 2.2. See what’s in voltage data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

for key, value in dc_data_dict.items():
    print(f"{key:<20}: {type(value)}")

######################################################################
#

dc_data_dict["a_locations"]

######################################################################
#

# error of voltages in percentage
err = np.hstack(dc_data_dict["standard_deviations"]) / np.hstack(dc_data_dict["observed_data"])
err *= 100
np.min(err), np.max(err)

######################################################################
#

dc_data_dict["standard_deviations"]

######################################################################
#


######################################################################
# 2.3. Create SimPEG Data and Survey
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

import matplotlib.pyplot as plt
import ipywidgets

from SimPEG import (
    Data, maps,
    data_misfit, regularization, optimization, inverse_problem, 
    inversion, directives
) 
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import induced_polarization as ip

######################################################################
#

# initialize an empty list for each 
source_list = []

for i in range(dc_data_dict["n_sources"]):
    
    # receiver electrode locations in 2D 
    m_locs = np.vstack([
        dc_data_dict["m_locations"][i], 
        np.zeros_like(dc_data_dict["m_locations"][i])
    ]).T
    n_locs = np.vstack([
        dc_data_dict["n_locations"][i],
        np.zeros_like(dc_data_dict["n_locations"][i])
    ]).T
    
    # construct the receiver object 
    receivers = dc.receivers.Dipole(locations_m=m_locs, locations_n=n_locs)
    
    # construct the source 
    source = dc.sources.Dipole(
        location_a=np.r_[dc_data_dict["a_locations"][i], 0.],
        location_b=np.r_[dc_data_dict["b_locations"][i], 0.],
        receiver_list=[receivers]
    )
    
    # append the new source to the source list
    source_list.append(source)


######################################################################
#

dc_survey = dc.Survey(source_list=source_list)

######################################################################
#

dc_data = Data(
    survey=dc_survey, 
    dobs=np.hstack(dc_data_dict["observed_data"]),
    standard_deviation=np.hstack(dc_data_dict["standard_deviations"])
)

######################################################################
#

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
dc.utils.plot_pseudosection(
    dc_data, data_type="potential",
    plot_type="contourf", data_locations=True, ax=ax,
)
ax.set_aspect(1.5)

######################################################################
#


######################################################################
# 2.4. Process into apparent resistivity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We’ve loaded measurements data in units of volts above. Now we translate
# them into apparent resistivity.
# 
# See `this SimPEG
# notebook <https://github.com/simpeg/transform-2020-simpeg/blob/main/1-century-dcip-inversion.ipynb>`__
# for more details.
# 

# plot psuedosection
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
dc.utils.plot_pseudosection(
    dc_data, data_type="apparent resistivity", 
    plot_type="contourf", data_locations=True, ax=ax, cbar_opts={"pad":0.25}
)
ax.set_aspect(1.5)  # some vertical exxageration
ax.set_title(f"DC: {line} Pseudosection")
ax.set_xlabel("Northing (m)");

######################################################################
#

apparent_resistivity = dc.utils.apparent_resistivity_from_voltage(dc_survey, dc_data.dobs)
apparent_resistivity_err = dc.utils.apparent_resistivity_from_voltage(dc_survey, dc_data.standard_deviation)
# apparent_resistivity_err = dc_data.standard_deviation
geometric_factor = dc.utils.geometric_factor(dc_survey)

######################################################################
#

apparent_resistivity_err

######################################################################
#

fig, ax = plt.subplots(1, 1)
out = ax.hist(np.log10(apparent_resistivity), bins=30)
ax.set_xlabel("log$_{10}(\\rho_a)$");

######################################################################
#

rho0 = np.median(apparent_resistivity)
rho0

######################################################################
#

len(apparent_resistivity)

######################################################################
#

apparent_resistivity[:10]

######################################################################
#


######################################################################
# 3. Apparent Chargeability
# -------------------------
# 
# 3.1. Load chargeability data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

ip_data_dict = read_dcip_data(ip_data_file)

######################################################################
#


######################################################################
# 3.2. See what’s in chargeability data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

for key, value in ip_data_dict.items():
    print(f"{key:<20}: {type(value)}")

######################################################################
#

ip_data_dict["a_locations"]

######################################################################
#

apparent_chargeability = np.hstack(ip_data_dict["observed_data"])
apparent_chargeability_err = np.hstack(ip_data_dict["standard_deviations"])

######################################################################
#


######################################################################
# 3.3. Create SimPEG Data and Survey
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# initialize an empty list for each 
source_list_ip = []

for i in range(ip_data_dict["n_sources"]):
    
    # receiver electrode locations in 2D 
    m_locs = np.vstack([
        ip_data_dict["m_locations"][i], 
        np.zeros_like(ip_data_dict["m_locations"][i])
    ]).T
    n_locs = np.vstack([
        ip_data_dict["n_locations"][i],
        np.zeros_like(ip_data_dict["n_locations"][i])
    ]).T
    
    # construct the receiver object 
    receivers = ip.receivers.Dipole(
        locations_m=m_locs, locations_n=n_locs, data_type="apparent_chargeability"
    )
    
    # construct the source 
    source = ip.sources.Dipole(
        location_a=np.r_[ip_data_dict["a_locations"][i], 0.],
        location_b=np.r_[ip_data_dict["b_locations"][i], 0.],
        receiver_list=[receivers]
    )
    
    # append the new source to the source list
    source_list_ip.append(source)

######################################################################
#

survey_ip = ip.Survey(source_list_ip)

######################################################################
#

ip_data = Data(
    survey=dc_survey, 
    dobs=np.hstack(ip_data_dict["observed_data"]), 
    standard_deviation=np.hstack(ip_data_dict["standard_deviations"])
)

######################################################################
#

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
dc.utils.plot_pseudosection(
    ip_data, data_type="potential",
    plot_type="contourf", data_locations=True, ax=ax,
)
ax.set_aspect(1.5)

######################################################################
#

len(ip_data.dobs)

######################################################################
#


######################################################################
# 4. Write data to file
# ---------------------
# 

processed_data = np.zeros((len(apparent_resistivity), 9))
processed_data[:,4] = apparent_resistivity
processed_data[:,5] = apparent_resistivity_err
processed_data[:,6] = apparent_chargeability
processed_data[:,7] = apparent_chargeability_err
processed_data[:,8] = geometric_factor

data_idx = 0
for i in range(len(dc_data_dict["a_locations"])):
    a_loc = dc_data_dict["a_locations"][i]
    b_loc = dc_data_dict["b_locations"][i]
    m_locs = dc_data_dict["m_locations"][i]
    n_locs = dc_data_dict["n_locations"][i]
    for j in range(len(m_locs)):
        processed_data[data_idx,0] = a_loc
        processed_data[data_idx,1] = b_loc
        processed_data[data_idx,2] = m_locs[j]
        processed_data[data_idx,3] = n_locs[j]
        data_idx += 1

processed_data[:5]

######################################################################
#

np.savetxt(
    "../century_dcip_data.txt", 
    processed_data, 
    header="# a_loc, b_loc, m_loc, n_loc, apparent_resistivity, apparent_resistivity_err, apparent_chargeability, apparent_chargeability_err, geometric_factor"
)

######################################################################
#


######################################################################
# 5. Convert mesh and model files format
# --------------------------------------
# 

import discretize

######################################################################
#

dc_model_file = f"./century/{line}/DCMODA.CON"
ip_model_file = f"./century/{line}/IPMODA.CHG"
mesh_file = f"./century/{line}/{line[:3]}MESH.DAT"

mesh_file, dc_model_file, ip_model_file

######################################################################
#

mesh_ubc = discretize.TensorMesh.read_UBC(mesh_file)
mesh_tensor = mesh_ubc.get_tensor("nodes")
np.savetxt("../century_mesh_nodes_x.txt", mesh_tensor[0])
np.savetxt("../century_mesh_nodes_z.txt", mesh_tensor[1])

######################################################################
#

def read_ubc_model(filename, mesh_ubc=mesh_ubc): 
    """
    A function to read a UBC conductivity or chargeability model. 
    """
    values = np.genfromtxt(
        filename, delimiter=' \n',
        dtype=str, comments='!', skip_header=1
    )
    tmp = np.hstack([np.array(value.split(), dtype=float) for value in values])
    model_ubc = discretize.utils.mkvc(tmp.reshape(mesh_ubc.vnC, order='F')[:,::-1])
    return model_ubc

# DC
sigma_ubc = read_ubc_model(dc_model_file)
rho_ubc = 1./sigma_ubc
np.savetxt("../century_dc_model.txt", rho_ubc)

# IP
charge_ubc = read_ubc_model(ip_model_file)
np.savetxt("../century_ip_model.txt", charge_ubc)

######################################################################
#


######################################################################
# --------------
# 
# Watermark
# ---------
# 

import SimPEG
SimPEG.__version__

######################################################################
#