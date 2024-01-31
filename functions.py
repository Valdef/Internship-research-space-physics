#!/usr/bin/env python
# coding: utf-8

import speasy as spz
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import pandas as pd 
import import_ipynb
import SerPyShock as SP

amda_tree = spz.inventories.tree.amda

cda_tree = spz.inventories.tree.cda

ssc_tree = spz.inventories.tree.ssc
    
#------ function to calculate the start and end time of the data taken ------
def find_time(day_shock, month_shock, year_shock, time_shock, time_period):
    if isinstance(time_shock, int):
        # Time is given in seconds
        hour_shock = int(time_shock // 3600)
        minute_shock = int((time_shock % 3600) // 60)
        second_shock = int(time_shock % 60)
    elif isinstance(time_shock, str):
        # Time is given in HH:MM:SS format
        hour_shock, minute_shock, second_shock = map(int, time_shock.split(':'))
    else:
        raise ValueError("Invalid time format. Please provide either seconds or HH:MM:SS.")

    # Define range to search the data and the time of the shock
    start_date = datetime(year_shock, month_shock, day_shock, hour_shock, minute_shock, second_shock) - timedelta(minutes=time_period)
    end_date = datetime(year_shock, month_shock, day_shock, hour_shock, minute_shock, second_shock) + timedelta(minutes=time_period)
    time_shock = datetime(year_shock, month_shock, day_shock, hour_shock, minute_shock, second_shock)

    dt = end_date - start_date

    return start_date, end_date, time_shock






# --------------- function to plot the line and windows time ----------------
def plot_line(time_up,time_down,up_window,down_window, up_shk, dw_shk, time_shock,ax):
    #------ averaging window for our calculation ------
    ax.axvspan(time_up[0],time_up[1], facecolor = "grey", alpha = 1)
    ax.axvspan(time_down[0],time_down[1], facecolor = "grey", alpha = 1)
    
    #------ averaging windows for the Trotta code -----
    ax.axvspan(up_shk, up_window[0], facecolor = "green", alpha = 0.6)
    ax.axvspan(up_shk, up_window[1], facecolor = "green", alpha = 0.3)
    ax.axvspan(dw_shk,down_window[0], facecolor = "red", alpha = 0.6)
    ax.axvspan(dw_shk,down_window[1], facecolor = "red", alpha = 0.3)
    ax.axvline(x = time_shock, color = "black")





# ---- define function to filtered Bup within the averaging window-----
def filtered_B_up(time_mag, Bx, By, Bz, time_up):
    filtered_Bx_up = []
    filtered_By_up = []
    filtered_Bz_up = []

    for timestamp, bx, by, bz in zip(time_mag, Bx, By, Bz):
        truncated_timestamp = str(timestamp)[:26]  # Truncate to 6-digit fractional seconds
        timestamp_datetime = datetime.strptime(truncated_timestamp, '%Y-%m-%dT%H:%M:%S.%f')
        if time_up[0] <= timestamp_datetime <= time_up[1]:
            filtered_Bx_up.append(bx)
            filtered_By_up.append(by)
            filtered_Bz_up.append(bz)
            
    return filtered_Bx_up, filtered_By_up, filtered_Bz_up





# ---- define function to filtered Bdown winthin the averaging window-----
from datetime import datetime

def filtered_B_down(time_mag, Bx, By, Bz, time_down):
    filtered_Bx_down = []
    filtered_By_down = []
    filtered_Bz_down = []

    for timestamp, bx, by, bz in zip(time_mag, Bx, By, Bz):
        truncated_timestamp = str(timestamp)[:26]  # Truncate to 6-digit fractional seconds
        timestamp_datetime = datetime.strptime(truncated_timestamp, '%Y-%m-%dT%H:%M:%S.%f')
        if time_down[0] <= timestamp_datetime <= time_down[1]:
            filtered_Bx_down.append(bx)
            filtered_By_down.append(by)
            filtered_Bz_down.append(bz)
            
    return filtered_Bx_down, filtered_By_down, filtered_Bz_down






# ---- define function to filtered Vup winthin the averaging window-----
def filtered_V_up(time_velo,Vx, Vy, Vz,time_up):
    # Filter Vx, Vy, Vz values within the desired time range
    filtered_Vx_up = []
    filtered_Vy_up = []
    filtered_Vz_up = []

    for timestamp, vx, vy, vz in zip(time_velo, Vx, Vy, Vz):
        timestamp_str = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        if time_up[0] <= timestamp_datetime <= time_up[1]:
            filtered_Vx_up.append(vx)
            filtered_Vy_up.append(vy)
            filtered_Vz_up.append(vz)
      
    return filtered_Vx_up, filtered_Vy_up, filtered_Vz_up






# ---- define function to filtered Vdown winthin the averaging window-----
def filtered_V_down(time_velo,Vx, Vy, Vz,time_down):
    # Filter Vx, Vy, Vz values within the desired time range
    filtered_Vx_down = []
    filtered_Vy_down = []
    filtered_Vz_down = []

    for timestamp, vx, vy, vz in zip(time_velo, Vx, Vy, Vz):
        timestamp_str2 = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str2, '%Y-%m-%dT%H:%M:%S')
        if time_down[0] <= timestamp_datetime <= time_down[1]:
            filtered_Vx_down.append(vx)
            filtered_Vy_down.append(vy)
            filtered_Vz_down.append(vz)
            
    return filtered_Vx_down, filtered_Vy_down, filtered_Vz_down







# ---- define function to filtered Nup winthin the averaging window-----
def filtered_N_up(time_rho,Nx,time_up):
    # Filter Nx, Ny, Nz values within the desired time range
    filtered_Nx_up = []

    for timestamp, nx in zip(time_rho, Nx):
        timestamp_str2 = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str2, '%Y-%m-%dT%H:%M:%S')
        if time_up[0] <= timestamp_datetime <= time_up[1]:
            filtered_Nx_up.append(nx)
            
    return filtered_Nx_up






# ---- define function to filtered Ndown winthin the averaging window-----
def filtered_N_down(time_rho,Nx,time_down):
    # Filter nx, ny, nz values within the desired time range
    filtered_Nx_down = []

    for timestamp, nx in zip(time_rho, Nx):
        timestamp_str2 = str(timestamp).split('.')[0]  # Convert to string and remove fractional seconds
        timestamp_datetime = datetime.strptime(timestamp_str2, '%Y-%m-%dT%H:%M:%S')
        if time_down[0] <= timestamp_datetime <= time_down[1]:
            filtered_Nx_down.append(nx)
            
    return filtered_Nx_down






# -------- create function to compare the different array --------
def compare_normal(array1,array2, accuracy):
    def compare_arrays(array1, array2, accuracy):
        if len(array1) != len(array2):
            return False

        for i in range(len(array1)):
            percentage_difference = abs(array1[i] - array2[i]) / abs(array2[i]) * 100
            if percentage_difference > accuracy:
                return False

        return True

    if compare_arrays(array1, array2, accuracy):
        print("The values are within ±" f"{accuracy}% accuracy.")
    else:
        print("The values are NOT within ±" f"{accuracy}% accuracy.")
        
    print("")


    
    
    
    
    
# function to compare values together -----    
def compare_float(value1,value2, accuracy):
    def compare_values(value1,value2, accuracy):
        percentage_difference = abs(value1 - value2) / abs(value2) * 100
        if percentage_difference > accuracy:
            return False

        return True
    
    if compare_values(value1, value2, accuracy):
        print("The values are within ±" f"{accuracy}% accuracy.")
    else:
        print("The values are NOT within ±" f"{accuracy}% accuracy.")
        
    print("")
    

    
    
    
    
# ------ function to calculate average of up and down ------
def calculate_average(filtered_x_up,filtered_x_down,filtered_y_up,filtered_y_down,filtered_z_up,filtered_z_down):    
    # Calculate the average of the filtered values
    x_up = np.nanmean(filtered_x_up)
    x_down = np.nanmean(filtered_x_down)

    y_up = np.nanmean(filtered_y_up)
    y_down = np.nanmean(filtered_y_down)

    z_up = np.nanmean(filtered_z_up)
    z_down = np.nanmean(filtered_z_down)

    return x_up, x_down, y_up, y_down, z_up, z_down
    
    
    


# -------- create function to calculate magnitude --------
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))







# ----- function to get the data of the magnetic field ------ 
def get_Bfield(spacecraft, start_date, end_date, time_up, time_down):
    #collect data from desired spacecraft;
    if spacecraft == 'C1':
        cluster1_B = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.FGM.clust1_fgm_high.c1_b_5vps, start_date, end_date)
        time_mag = cluster1_B.time
        B = cluster1_B.values
        B_unit = cluster1_B.unit
        
        Bx = cluster1_B.filter_columns([]).values
        By = cluster1_B.filter_columns([]).values
        Bz = cluster1_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.FGM.clust1_fgm_high.c1_b_5vps, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.FGM.clust1_fgm_high.c1_b_5vps, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'C3':
        cluster3_B = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.FGM.clust3_fgm_high.c3_b_5vps, start_date, end_date)
        time_mag = cluster3_B.time
        B = cluster3_B.values
        B_unit = cluster3_B.unit
        
        Bx = cluster3_B.filter_columns([]).values
        By = cluster3_B.filter_columns([]).values
        Bz = cluster3_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.FGM.clust3_fgm_high.c3_b_5vps, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.FGM.clust3_fgm_high.c3_b_5vps, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'C4':
        cluster4_B = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.FGM.clust4_fgm_high.c4_b_5vps, start_date, end_date)
        time_mag = cluster4_B.time
        B = cluster4_B.values
        B_unit = cluster4_B.unit
        
        Bx = cluster4_B.filter_columns([]).values
        By = cluster4_B.filter_columns([]).values
        Bz = cluster4_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.FGM.clust4_fgm_high.c4_b_5vps, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.FGM.clust4_fgm_high.c4_b_5vps, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    
    if spacecraft == 'DS1':
        DS1_B = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.FGM.dstar1_fgm_prp.ds1_b, start_date, end_date)
        time_mag = DS1_B.time
        B = DS1_B.values
        B_unit = DS1_B.unit
        
        Bx = DS1_B.filter_columns([]).values
        By = DS1_B.filter_columns([]).values
        Bz = DS1_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.FGM.dstar1_fgm_prp.ds1_b, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.FGM.dstar1_fgm_prp.ds1_b, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    
    if spacecraft == 'THA':
        THA_B = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.FGM.tha_fgm_s.tha_bs, start_date, end_date)
        time_mag = THA_B.time
        B = THA_B.values
        B_unit = THA_B.unit
        
        Bx = THA_B.filter_columns([]).values
        By = THA_B.filter_columns([]).values
        Bz = THA_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.FGM.tha_fgm_s.tha_bs, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.FGM.tha_fgm_s.tha_bs, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'THB':
        THB_B = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.FGM.thb_fgm_s.thb_bs, start_date, end_date)
        time_mag = THB_B.time
        B = THB_B.values
        B_unit = THB_B.unit
        
        Bx = THB_B.filter_columns([]).values
        By = THB_B.filter_columns([]).values
        Bz = THB_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.FGM.thb_fgm_s.thb_bs, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.FGM.thb_fgm_s.thb_bs, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'THC':
        THC_B = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.FGM.thc_fgm_s.thc_bs, start_date, end_date)
        time_mag = THC_B.time
        B = THC_B.values
        B_unit = THC_B.unit
        
        Bx = THC_B.filter_columns([]).values
        By = THC_B.filter_columns([]).values
        Bz = THC_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.FGM.thc_fgm_s.thc_bs, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.FGM.thc_fgm_s.thc_bs, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'THD':
        THD_B = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.FGM.thd_fgm_s.thd_bs, start_date, end_date)
        time_mag = THD_B.time
        B = THD_B.values
        B_unit = THD_B.unit
        
        Bx = THD_B.filter_columns([]).values
        By = THD_B.filter_columns([]).values
        Bz = THD_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.FGM.thd_fgm_s.thd_bs, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.FGM.thd_fgm_s.thd_bs, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'THE':
        THE_B = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.FGM.the_fgm_s.the_bs, start_date, end_date)
        time_mag = THE_B.time
        B = THE_B.values
        B_unit = THE_B.unit
        
        Bx = THE_B.filter_columns([]).values
        By = THE_B.filter_columns([]).values
        Bz = THE_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.FGM.the_fgm_s.the_bs, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.FGM.the_fgm_s.the_bs, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'MMS':
        MMS_B = spz.get_data(amda_tree.Parameters.MMS.MMS1.FGM.mms1_fgm_srvy.mms1_b_gse, start_date, end_date)
        time_mag = MMS_B.time
        B = MMS_B.values
        B_unit = MMS_B.unit
        
        Bx = MMS_B.filter_columns([]).values
        By = MMS_B.filter_columns([]).values
        Bz = MMS_B.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.MMS.MMS1.FGM.mms1_fgm_srvy.mms1_b_gse, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.MMS.MMS1.FGM.mms1_fgm_srvy.mms1_b_gse, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'WIND':
        wind_mag = spz.get_data(amda_tree.Parameters.Wind.MFI.wnd_mfi_high.wnd_bh, start_date, end_date)
        time_mag = wind_mag.time
        B = wind_mag.values
        B_unit = wind_mag.unit
        
        Bx = wind_mag.filter_columns([]).values
        By = wind_mag.filter_columns([]).values
        Bz = wind_mag.filter_columns([]).values
        
        B_up = spz.get_data(amda_tree.Parameters.Wind.MFI.wnd_mfi_high.wnd_bh, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Wind.MFI.wnd_mfi_high.wnd_bh, time_down[0], time_down[1])
        
        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'INTERBALL':
        Interball_B = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.MIF_M.iball_mif_hr.it_b_gse, start_date, end_date)
        time_mag = Interball_B.time
        B = Interball_B.values
        B_unit = Interball_B.unit

        Bx = Interball_B.filter_columns([]).values
        By = Interball_B.filter_columns([]).values
        Bz = Interball_B.filter_columns([]).values

        B_up = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.MIF_M.iball_mif_hr.it_b_gse, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.MIF_M.iball_mif_hr.it_b_gse, time_down[0], time_down[1])

        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    
    if spacecraft == 'GEOTAIL':
        Geotail_B = spz.get_data(amda_tree.Parameters.Geotail.MGF.gtl_mgf_edb.gtl_b_edb, start_date, end_date)
        time_mag = Geotail_B.time
        B = Geotail_B.values
        B_unit = Geotail_B.unit

        Bx = Geotail_B.filter_columns([]).values
        By = Geotail_B.filter_columns([]).values
        Bz = Geotail_B.filter_columns([]).values

        B_up = spz.get_data(amda_tree.Parameters.Geotail.MGF.gtl_mgf_edb.gtl_b_edb, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Geotail.MGF.gtl_mgf_edb.gtl_b_edb, time_down[0], time_down[1])

        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down

    
    else: 
        
        print('Function failed: spacecraft not recognised')
        




        

# ----- function to get velocity on any satellite -------
def get_velocity(spacecraft, start_date, end_date, time_up, time_down):

    # collect data from desired spacecraft;
    if spacecraft == 'C1':
        cluster1_v = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_v, start_date, end_date)

        time_velo = cluster1_v.time
        V = cluster1_v.values
        V_unit = cluster1_v.unit
        
        #splits data into GSE coordinates
        Vx = cluster1_v.filter_columns([]).values
        Vy = cluster1_v.filter_columns([]).values
        Vz = cluster1_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_v, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_v, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'C3':
        cluster3_v = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_v, start_date, end_date)

        time_velo = cluster3_v.time
        V = cluster3_v.values
        V_unit = cluster3_v.unit
        
        Vx = cluster3_v.filter_columns([]).values
        Vy = cluster3_v.filter_columns([]).values
        Vz = cluster3_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_v, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_v, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'C4':
        cluster4_v = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_v, start_date, end_date)

        time_velo = cluster4_v.time
        V = cluster4_v.values
        V_unit = cluster4_v.unit
        
        Vx = cluster4_v.filter_columns([]).values
        Vy = cluster4_v.filter_columns([]).values
        Vz = cluster4_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_v, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_v, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
     
    
    if spacecraft == 'DS1':
        DS1_v = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.HIA.dstar1_hia_prp.ds1_v, start_date, end_date)

        time_velo = DS1_v.time
        V = DS1_v.values
        V_unit = DS1_v.unit
        
        Vx = DS1_v.filter_columns([]).values
        Vy = DS1_v.filter_columns([]).values
        Vz = DS1_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.HIA.dstar1_hia_prp.ds1_v, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.HIA.dstar1_hia_prp.ds1_v, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
        
    if spacecraft == 'THA':
        THA_v = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.ESA.tha_peim_all.tha_v_peim, start_date, end_date)
        
        time_velo = THA_v.time
        V = THA_v.values
        V_unit = THA_v.unit
        
        Vx = THA_v.filter_columns([]).values
        Vy = THA_v.filter_columns([]).values
        Vz = THA_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.ESA.tha_peim_all.tha_v_peim, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.ESA.tha_peim_all.tha_v_peim, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'THB':
        THB_v = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.ESA.thb_peim_all.thb_v_peim, start_date, end_date)
        
        time_velo = THB_v.time
        V = THB_v.values
        V_unit = THB_v.unit
        
        Vx = THB_v.filter_columns([]).values
        Vy = THB_v.filter_columns([]).values
        Vz = THB_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.ESA.thb_peim_all.thb_v_peim, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.ESA.thb_peim_all.thb_v_peim, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'THC':
        THC_v = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.ESA.thc_peim_all.thc_v_peim, start_date, end_date)
        
        time_velo = THC_v.time
        V = THC_v.values
        V_unit = THC_v.unit
        
        Vx = THC_v.filter_columns([]).values
        Vy = THC_v.filter_columns([]).values
        Vz = THC_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.ESA.thc_peim_all.thc_v_peim, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.ESA.thc_peim_all.thc_v_peim, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'THD':
        THD_v = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.ESA.thd_peim_all.thd_v_peim, start_date, end_date)
        
        time_velo = THD_v.time
        V = THD_v.values
        V_unit = THD_v.unit
        
        Vx = THD_v.filter_columns([]).values
        Vy = THD_v.filter_columns([]).values
        Vz = THD_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.ESA.thd_peim_all.thd_v_peim, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.ESA.thd_peim_all.thd_v_peim, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'THE':
        THE_v = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.ESA.the_peim_all.the_v_peim, start_date, end_date)
        
        time_velo = THE_v.time
        V = THE_v.values
        V_unit = THE_v.unit
        
        Vx = THE_v.filter_columns([]).values
        Vy = THE_v.filter_columns([]).values
        Vz = THE_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.ESA.the_peim_all.the_v_peim, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.ESA.the_peim_all.the_v_peim, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'MMS':
        MMS_v = spz.get_data(amda_tree.Parameters.MMS.MMS1.FPI.fast_mode.mms1_fpi_dismoms.mms1_dis_vgse, start_date, end_date)

        time_velo = MMS_v.time
        V = MMS_v.values
        V_unit = MMS_v.unit
        
        Vx = MMS_v.filter_columns([]).values
        Vy = MMS_v.filter_columns([]).values
        Vz = MMS_v.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.MMS.MMS1.HPCA.mms1_hpca_moms.mms1_heplus_vgse, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.MMS.MMS1.HPCA.mms1_hpca_moms.mms1_heplus_vgse, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'WIND':
        wind_velo = spz.get_data(amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_v, start_date, end_date)
        
        time_velo = wind_velo.time
        V = wind_velo.values
        V_unit = wind_velo.unit
        
        Vx = wind_velo.filter_columns([]).values
        Vy = wind_velo.filter_columns([]).values
        Vz = wind_velo.filter_columns([]).values
        
        V_up = spz.get_data(amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_v, time_up[0], time_up[1])
        V_down = spz.get_data(amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_v, time_down[0], time_down[1])
        
        return time_velo, V, V_unit, Vx, Vy, Vz, V_up, V_down
    
    if spacecraft == 'INTERBALL':
        Interball_velo = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Corall.intt_crl_kp.it_v_gse, start_date, end_date)
        time_mag = Interball_velo.time
        B = Interball_velo.values
        B_unit = Interball_velo.unit

        Bx = Interball_velo.filter_columns([]).values
        By = Interball_velo.filter_columns([]).values
        Bz = Interball_velo.filter_columns([]).values

        B_up = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Corall.intt_crl_kp.it_v_gse, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Corall.intt_crl_kp.it_v_gse, time_down[0], time_down[1])

        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    if spacecraft == 'GEOTAIL':
        Geotail_velo = spz.get_data(amda_tree.Parameters.Geotail.LEP.gtl_lep_edb.gtl_lepb_v, start_date, end_date)
        time_mag = Geotail_velo.time
        B = Geotail_velo.values
        B_unit = Geotail_velo.unit

        Bx = Geotail_velo.filter_columns([]).values
        By = Geotail_velo.filter_columns([]).values
        Bz = Geotail_velo.filter_columns([]).values

        B_up = spz.get_data(amda_tree.Parameters.Geotail.LEP.gtl_lep_edb.gtl_lepb_v, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Geotail.LEP.gtl_lep_edb.gtl_lepb_v, time_down[0], time_down[1])

        return time_mag, B, B_unit, Bx, By, Bz, B_up, B_down
    
    else: 
        
        print('Function failed: spacecraft not recognised')
        

        
        
        
        
# --------- function to get density on any satellite ---------------
def get_density(spacecraft, start_date, end_date, time_up, time_down):

    #Collect data from desired spacecraft;
    if spacecraft == 'C1':
        cluster1_density = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_dens, start_date, end_date)
        
        time_rho = cluster1_density.time
        density = cluster1_density.values
        density_unit = cluster1_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_dens, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_dens, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'C3':
        cluster3_density = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_dens, start_date, end_date)
        
        time_rho = cluster3_density.time
        density = cluster3_density.values
        density_unit = cluster3_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_dens, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.CIS_HIA.clust3_hia_mom.c3_hia_dens, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'C4':
        cluster4_density = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_dens, start_date, end_date)
        
        time_rho = cluster4_density.time
        density = cluster4_density.values
        density_unit = cluster4_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_dens, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.CIS_HIA.clust4_hia_mom.c4_hia_dens, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    
    if spacecraft == 'DS1':
        DS1_density = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.HIA.dstar1_hia_prp.ds1_n, start_date, end_date)
        
        time_rho = DS1_density.time
        density = DS1_density.values
        density_unit = DS1_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.HIA.dstar1_hia_prp.ds1_n, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.HIA.dstar1_hia_prp.ds1_n, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    
    if spacecraft == 'THA':
        THA_density = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.ESA.tha_peim_all.tha_n_peim, start_date, end_date)
        
        time_rho = THA_density.time
        density = THA_density.values
        density_unit = THA_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.ESA.tha_peim_all.tha_n_peim, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.ESA.tha_peim_all.tha_n_peim, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'THB':
        THB_density = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.ESA.thb_peim_all.thb_n_peim, start_date, end_date)
        
        time_rho = THB_density.time
        density = THB_density.values
        density_unit = THB_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.ESA.thb_peim_all.thb_n_peim, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.ESA.thb_peim_all.thb_n_peim, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'THC':
        THC_density = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.ESA.thc_peim_all.thc_n_peim, start_date, end_date)
        
        time_rho = THC_density.time
        density = THC_density.values
        density_unit = THC_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.ESA.thc_peim_all.thc_n_peim, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.ESA.thc_peim_all.thc_n_peim, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'THD':
        THD_density = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.ESA.thd_peim_all.thd_n_peim, start_date, end_date)
        
        time_rho = THD_density.time
        density = THD_density.values
        density_unit = THD_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.ESA.thd_peim_all.thd_n_peim, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.ESA.thd_peim_all.thd_n_peim, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'THE':
        THE_density = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.ESA.the_peim_all.the_n_peim, start_date, end_date)
        
        time_rho = THE_density.time
        density = THE_density.values
        density_unit = THE_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.ESA.the_peim_all.the_n_peim, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.ESA.the_peim_all.the_n_peim, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'MMS':
        MMS_density = spz.get_data(amda_tree.Parameters.MMS.MMS1.FPI.fast_mode.mms1_fpi_dismoms.mms1_dis_ni, start_date, end_date)
        
        time_rho = MMS_density.time
        density = MMS_density.values
        density_unit = MMS_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.MMS.MMS1.FPI.fast_mode.mms1_fpi_dismoms.mms1_dis_ni, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.MMS.MMS1.FPI.fast_mode.mms1_fpi_dismoms.mms1_dis_ni, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'WIND':
        WIND_density = spz.get_data(amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_n, start_date, end_date)
        
        time_rho = WIND_density.time
        density = WIND_density.values
        density_unit = WIND_density.unit
        
        density_up = spz.get_data(amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_n, time_up[0], time_up[1])
        density_down = spz.get_data(amda_tree.Parameters.Wind.SWE.wnd_swe_kp.wnd_swe_n, time_down[0], time_down[1])
        
        return time_rho, density, density_unit, density_up, density_down
    
    if spacecraft == 'INTERBALL':
        Interball_density = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Corall.intt_crl_kp.it_n, start_date, end_date)
        time_mag = Interball_density.time
        B = Interball_density.values
        B_unit = Interball_density.unit

        B_up = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Corall.intt_crl_kp.it_n, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Corall.intt_crl_kp.it_n, time_down[0], time_down[1])

        return time_mag, B, B_unit, B_up, B_down
    
    if spacecraft == 'GEOTAIL':
        Geotail_density = spz.get_data(amda_tree.Parameters.Geotail.LEP.gtl_lep_edb.gtl_lepb_dens, start_date, end_date)
        time_mag = Geotail_density.time
        B = Geotail_density.values
        B_unit = Geotail_density.unit

        B_up = spz.get_data(amda_tree.Parameters.Geotail.LEP.gtl_lep_edb.gtl_lepb_dens, time_up[0], time_up[1])
        B_down = spz.get_data(amda_tree.Parameters.Geotail.LEP.gtl_lep_edb.gtl_lepb_dens, time_down[0], time_down[1])

        return time_mag, B, B_unit, B_up, B_down
    
    else: 
        
        print('Function failed: spacecraft not recognised')
        
        
#----------- function to get the coordinates of the satellites ----------

def get_coordinate(spacecraft,time_shock):
    
    if spacecraft == 'C1':
        orbit_c1 = spz.get_data(amda_tree.Parameters.Cluster.Cluster_1.Ephemeris.clust1_orb_all.c1_xyz_gse,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_c1.time
        orbit = orbit_c1.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft =='C3':
        orbit_c3 = spz.get_data(amda_tree.Parameters.Cluster.Cluster_3.Ephemeris.clust3_orb_all.c3_xyz_gse,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_c3.time
        orbit = orbit_c3.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit
   
    if spacecraft =='C4':
        orbit_c4 = spz.get_data(amda_tree.Parameters.Cluster.Cluster_4.Ephemeris.clust4_orb_all.c4_xyz_gse,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_c4.time
        orbit = orbit_c4.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'DS1':
        orbit_ds1 = spz.get_data(amda_tree.Parameters.Double_Star.DoubleStar_1.Ephemeris.dstar1_orb_all.ds1_xyz,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_ds1.time
        orbit = orbit_ds1.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'THA':
        orbit_tha = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_A.Ephemeris.tha_orb_all.tha_xyz,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_tha.time
        orbit = orbit_tha.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit
    
    if spacecraft == 'THB':
        orbit_thb = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_B.Ephemeris.thb_orb_all.thb_xyz,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_thb.time
        orbit = orbit_thb.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'THC':
        orbit_thc = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_C.Ephemeris.thc_orb_all.thc_xyz,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_thc.time
        orbit = orbit_thc.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'THD':
        orbit_thd = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_D.Ephemeris.thd_orb_all.thd_xyz,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_thd.time
        orbit = orbit_thd.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'THE':
        orbit_the = spz.get_data(amda_tree.Parameters.THEMIS.THEMIS_E.Ephemeris.the_orb_all.the_xyz,time_shock-timedelta(seconds=30), time_shock+timedelta(seconds=30))

        time_orbit = orbit_the.time
        orbit = orbit_the.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'MMS':
        orbit_mms = spz.get_data(amda_tree.Parameters.MMS.MMS1.Ephemeris.mms1_orb_t89d.mms1_xyz_gse,time_shock-timedelta(seconds=15), time_shock+timedelta(seconds=15))

        time_orbit = orbit_mms.time
        orbit = orbit_mms.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'WIND':
        orbit_wind = spz.get_data(amda_tree.Parameters.Wind.Ephemeris.wnd_orb_all.wnd_xyz_gse,time_shock-timedelta(minutes=5), time_shock+timedelta(minutes=5))

        time_orbit = orbit_wind.time
        orbit = orbit_wind.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit

    if spacecraft == 'INTERBALL':
        orbit_inter = spz.get_data(amda_tree.Parameters.Interball.Interball_Tail.Ephemeris.intt_orb_all.it_xyz,time_shock-timedelta(minutes=1), time_shock+timedelta(seconds=1))

        time_orbit = orbit_inter.time
        orbit = orbit_inter.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)

        return orbit, time_orbit
    
    if spacecraft == 'GEOTAIL':
        orbit_geo = spz.get_data(amda_tree.Parameters.Geotail.Ephemeris.gtl_orb_all.gtl_xyz,time_shock-timedelta(minutes=5), time_shock+timedelta(minutes=5))

        time_orbit = orbit_geo.time
        orbit = orbit_geo.values

        print("the time is:",time_orbit)
        print("the satellite coordinate are:", orbit)
        
        return orbit, time_orbit
        
    else: 
        print('Function failed: spacecraft not recognised')    