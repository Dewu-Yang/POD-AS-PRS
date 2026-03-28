#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata

def calculate_drag_coefficient(coords, velocity, pressure, metadata):
    print("[DEBUG] Using calculate_drag_coefficient from calculate_drag.py")
    """
    计算圆柱的阻力系数
    
    参考了DFG 2D-3基准测试中的公式:
    C_D = (2/L) * ∫_Γ_S [ν*(∂u_t/∂n)*n_y - p*n_x] ds
    其中L为特征长度
    
    参数:
    coords - 坐标数据
    velocity - 速度场数据
    pressure - 压力场数据
    metadata - 元数据包含必要的信息
    
    返回:
    c_d - 阻力系数
    """
    # 参数设置
    rho = 1.0            # 密度
    nu = 0.01           # 运动粘度
    D = 1.0              # 圆柱直径 (假设为1.0)
    U_mean = 1.0         # 参考速度
    
    # 创建圆柱表面的离散点 (减少点数以提高速度)
    n_points = 150  # 减少点数以加快计算速度
    theta = np.linspace(0, 2*np.pi, n_points)
    cylinder_x = 0.5 * np.cos(theta)  # 圆柱半径为0.5
    cylinder_y = 0.5 * np.sin(theta)
    
    # 计算每个点的法向量 (指向圆柱外部的单位法向量)
    nx = np.cos(theta)
    ny = np.sin(theta)
    
    # 展平坐标和场数据以用于插值
    x_coords = coords[:, 0, :].flatten()
    y_coords = coords[:, 1, :].flatten()
    u_component = velocity[:, 0, :].flatten()
    v_component = velocity[:, 1, :].flatten()
    pressure_values = pressure.flatten()
    
    # 创建圆柱附近的高分辨率网格
    # 只选择圆柱附近的点，使用更严格的筛选以减少数据点
    # 选择较小的区域以加快计算速度
    cylinder_vicinity_mask = ((x_coords >= -1.0) & (x_coords <= 1.0) & 
                             (y_coords >= -1.0) & (y_coords <= 1.0))
    
    vicinity_x = x_coords[cylinder_vicinity_mask]
    vicinity_y = y_coords[cylinder_vicinity_mask]
    vicinity_points = np.column_stack((vicinity_x, vicinity_y))
    vicinity_u = u_component[cylinder_vicinity_mask]
    vicinity_v = v_component[cylinder_vicinity_mask]
    vicinity_p = pressure_values[cylinder_vicinity_mask]
    
    # 使用快速的LinearNDInterpolator来加速计算
    print(f"Creating fast linear interpolator for cylinder vicinity using {len(vicinity_points)} data points")

    # 直接使用原始数据点建立插值器，不进行异常值过滤以加快速度
    u_interp = LinearNDInterpolator(vicinity_points, vicinity_u)
    v_interp = LinearNDInterpolator(vicinity_points, vicinity_v)
    p_interp = LinearNDInterpolator(vicinity_points, vicinity_p)
    
    # Initialize the drag integrand array
    drag_integrand = np.zeros(n_points)
    
    # Set up the second-order finite difference parameters (using a simpler difference method)
    # 使用最简单的三点差分模式加快计算
    distances = np.array([0.0, 0.005, 0.01])  # 只使用三个点进行差分
    # 使用二阶前向差分系数
    fd_coeffs = np.array([-3, 4, -1]) / (2 * distances[1])
    
    # Precompute the positions of all points for batch interpolation
    # 准备存储结果的数组
    u_values = np.zeros((n_points, len(distances)))
    v_values = np.zeros((n_points, len(distances)))
    p_values = np.zeros(n_points)
    
    # 计算表面上的压力和外部点的速度
    for i in range(n_points):
        x_s = cylinder_x[i]
        y_s = cylinder_y[i]
        
        # 获取表面点的压力
        p_values[i] = p_interp(x_s, y_s)
        
        # 计算用于差分的点的速度
        for j, dist in enumerate(distances):
            x_out = x_s + dist * nx[i]
            y_out = y_s + dist * ny[i]
            
            # 对于表面点(dist=0)，速度为零(no-slip条件)
            if dist == 0:
                u_values[i, j] = 0.0
                v_values[i, j] = 0.0
            else:
                # 对速度使用CloughTocher2D插值
                u_values[i, j] = u_interp(x_out, y_out)
                v_values[i, j] = v_interp(x_out, y_out)
    
    # 简化处理NaN值的方法，加快计算
    if np.isnan(u_values).any() or np.isnan(v_values).any() or np.isnan(p_values).any():
        print("警告: 插值中发现NaN值，使用最近邻插值修复")
        
        # 创建坐标点网格用于nearest插值
        pts = np.column_stack((vicinity_x, vicinity_y))
        
        # 修复NaN值
        for i in range(n_points):
            for j in range(len(distances)):
                # 对速度中的NaN处理
                if distances[j] == 0:
                    # 表面点总是零
                    if np.isnan(u_values[i, j]):
                        u_values[i, j] = 0.0
                    if np.isnan(v_values[i, j]):
                        v_values[i, j] = 0.0
                else:
                    # 非表面点的NaN值使用最近邻插值替换
                    x_out = cylinder_x[i] + distances[j] * nx[i]
                    y_out = cylinder_y[i] + distances[j] * ny[i]
                    if np.isnan(u_values[i, j]):
                        u_values[i, j] = griddata(pts, vicinity_u, np.array([[x_out, y_out]]), method='nearest')[0]
                    if np.isnan(v_values[i, j]):
                        v_values[i, j] = griddata(pts, vicinity_v, np.array([[x_out, y_out]]), method='nearest')[0]
            
            # 修复压力值
            if np.isnan(p_values[i]):
                x_s = cylinder_x[i]
                y_s = cylinder_y[i]
                p_values[i] = griddata(pts, vicinity_p, np.array([[x_s, y_s]]), method='nearest')[0]
    
    # Calculate the drag integrand at each point on the cylinder
    for i in range(n_points):
        # 切向单位向量
        tx = -ny[i]
        ty = nx[i]
        
        # 获取表面点处的压力
        p_s = p_values[i]
        
        # 计算所有点的切向速度分量
        u_t_values = np.zeros(len(distances))
        
        # 对每个距离点计算切向速度
        for j in range(len(distances)):
            # 表面点的速度为零(no-slip条件)
            if distances[j] == 0:
                u_t_values[j] = 0.0
            else:
                # 计算切向速度分量: u_t = u·t = u*tx + v*ty
                u_t_values[j] = u_values[i, j] * tx + v_values[i, j] * ty
        
        # 使用四阶有限差分计算法向导数: du_t/dn
        du_t_dn = np.sum(fd_coeffs * u_t_values)
        
        # 阻力被积函数: ν*(∂u_t/∂n)*n_y - p*n_x
        drag_integrand[i] = nu * du_t_dn * ny[i] - p_s * nx[i]
    
    # Integrate using the trapezoidal rule (simpler than Simpson's rule and sufficient for accuracy)
    # 使用简单的梯形法则加快计算
    ds = 2 * np.pi / n_points  # 角度步长
    
    # 简单平滑处理，去除特异值
    mean_val = np.mean(drag_integrand)
    std_val = np.std(drag_integrand)
    for i in range(n_points):
        if np.abs(drag_integrand[i] - mean_val) > 2.5 * std_val:
            # 特异值替换为相邻点平均值
            if i > 0 and i < n_points-1:
                drag_integrand[i] = (drag_integrand[i-1] + drag_integrand[i+1]) / 2
            elif i == 0:
                drag_integrand[i] = (drag_integrand[n_points-1] + drag_integrand[i+1]) / 2
            else:
                drag_integrand[i] = (drag_integrand[i-1] + drag_integrand[0]) / 2
    
    # 使用梯形积分法则
    drag_force = 0.5 * np.sum(drag_integrand[:-1] + drag_integrand[1:]) * ds
    
    # Calculate the drag coefficient
    c_d = (2 / D) * drag_force / (rho * U_mean**2)
    
    return c_d
