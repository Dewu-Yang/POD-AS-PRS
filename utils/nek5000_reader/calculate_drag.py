#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata

def calculate_drag_coefficient(coords, velocity, pressure, metadata):
    print("[DEBUG] Using calculate_drag_coefficient from calculate_drag.py")
    
    rho = 1.0            
    nu = 0.01           
    D = 1.0              
    U_mean = 1.0         
    
    n_points = 150  
    theta = np.linspace(0, 2*np.pi, n_points)
    cylinder_x = 0.5 * np.cos(theta)  
    cylinder_y = 0.5 * np.sin(theta)
    
    nx = np.cos(theta)
    ny = np.sin(theta)
    
    x_coords = coords[:, 0, :].flatten()
    y_coords = coords[:, 1, :].flatten()
    u_component = velocity[:, 0, :].flatten()
    v_component = velocity[:, 1, :].flatten()
    pressure_values = pressure.flatten()
    
    cylinder_vicinity_mask = ((x_coords >= -1.0) & (x_coords <= 1.0) & 
                             (y_coords >= -1.0) & (y_coords <= 1.0))
    
    vicinity_x = x_coords[cylinder_vicinity_mask]
    vicinity_y = y_coords[cylinder_vicinity_mask]
    vicinity_points = np.column_stack((vicinity_x, vicinity_y))
    vicinity_u = u_component[cylinder_vicinity_mask]
    vicinity_v = v_component[cylinder_vicinity_mask]
    vicinity_p = pressure_values[cylinder_vicinity_mask]
    
    print(f"Creating fast linear interpolator for cylinder vicinity using {len(vicinity_points)} data points")

    u_interp = LinearNDInterpolator(vicinity_points, vicinity_u)
    v_interp = LinearNDInterpolator(vicinity_points, vicinity_v)
    p_interp = LinearNDInterpolator(vicinity_points, vicinity_p)
    
    drag_integrand = np.zeros(n_points)
    
    distances = np.array([0.0, 0.005, 0.01])
    fd_coeffs = np.array([-3, 4, -1]) / (2 * distances[1])
    
    u_values = np.zeros((n_points, len(distances)))
    v_values = np.zeros((n_points, len(distances)))
    p_values = np.zeros(n_points)
    
    for i in range(n_points):
        x_s = cylinder_x[i]
        y_s = cylinder_y[i]
        
        p_values[i] = p_interp(x_s, y_s)
        
        for j, dist in enumerate(distances):
            x_out = x_s + dist * nx[i]
            y_out = y_s + dist * ny[i]
            
            if dist == 0:
                u_values[i, j] = 0.0
                v_values[i, j] = 0.0
            else:
                u_values[i, j] = u_interp(x_out, y_out)
                v_values[i, j] = v_interp(x_out, y_out)
    
    if np.isnan(u_values).any() or np.isnan(v_values).any() or np.isnan(p_values).any():
        print("Warning: NaN values found in interpolation, using nearest neighbor interpolation")
        
        pts = np.column_stack((vicinity_x, vicinity_y))
        
        for i in range(n_points):
            for j in range(len(distances)):
                if distances[j] == 0:
                    if np.isnan(u_values[i, j]):
                        u_values[i, j] = 0.0
                    if np.isnan(v_values[i, j]):
                        v_values[i, j] = 0.0
                else:
                    x_out = cylinder_x[i] + distances[j] * nx[i]
                    y_out = cylinder_y[i] + distances[j] * ny[i]
                    if np.isnan(u_values[i, j]):
                        u_values[i, j] = griddata(pts, vicinity_u, np.array([[x_out, y_out]]), method='nearest')[0]
                    if np.isnan(v_values[i, j]):
                        v_values[i, j] = griddata(pts, vicinity_v, np.array([[x_out, y_out]]), method='nearest')[0]
            
            if np.isnan(p_values[i]):
                x_s = cylinder_x[i]
                y_s = cylinder_y[i]
                p_values[i] = griddata(pts, vicinity_p, np.array([[x_s, y_s]]), method='nearest')[0]
    
    for i in range(n_points):
        tx = -ny[i]
        ty = nx[i]
        
        p_s = p_values[i]
        
        u_t_values = np.zeros(len(distances))
        
        for j in range(len(distances)):
            if distances[j] == 0:
                u_t_values[j] = 0.0
            else:
                u_t_values[j] = u_values[i, j] * tx + v_values[i, j] * ty
        
        du_t_dn = np.sum(fd_coeffs * u_t_values)
        
        drag_integrand[i] = nu * du_t_dn * ny[i] - p_s * nx[i]
    
    ds = 2 * np.pi / n_points
    
    mean_val = np.mean(drag_integrand)
    std_val = np.std(drag_integrand)
    for i in range(n_points):
        if np.abs(drag_integrand[i] - mean_val) > 2.5 * std_val:
            if i > 0 and i < n_points-1:
                drag_integrand[i] = (drag_integrand[i-1] + drag_integrand[i+1]) / 2
            elif i == 0:
                drag_integrand[i] = (drag_integrand[n_points-1] + drag_integrand[i+1]) / 2
            else:
                drag_integrand[i] = (drag_integrand[i-1] + drag_integrand[0]) / 2
    
    drag_force = 0.5 * np.sum(drag_integrand[:-1] + drag_integrand[1:]) * ds
    
    c_d = (2 / D) * drag_force / (rho * U_mean**2)
    
    return c_d
