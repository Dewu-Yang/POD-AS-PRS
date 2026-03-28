from pymech.neksuite import readnek
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置非交互式后端
import matplotlib.pyplot as plt

# 已成功加载
field = readnek('/mnt/HD4/NekExamples/ext_cyl/ext_cyl0.f00001')

# 访问数据
print(f"总元素数: {field.nel}")
print(f"时间步: {field.istep}")
print(f"物理时间: {field.time}")

print(field)

print([attr for attr in dir(field) if not attr.startswith('__')])
print(field.endian, field.istep, field.lr1, field.nbc, field.ncurv, field.ndim, field.nel, field.time, field.var, field.wdsz)
first_element = field.elem[0]

print([attr for attr in dir(first_element) if not attr.startswith('__')])
print(first_element.bcs, first_element.ccurv, first_element.curv)
print("Shape of element velocity and pressure arrays = ", first_element.vel.shape, first_element.pres.shape)

from fld_data import FldData

f = FldData.fromfile('/mnt/HD4/NekExamples/ext_cyl/ext_cyl0.f00125')
print(f)
print(f'流场维度: {f.ndims}')

# print(f'压力：{f.p}')
print(f'压力维度: {f.p.shape}')

# print(f'速度：{f.u}')
print(f'速度维度: {f.u.shape}')

# 速度的维度是: (1472, 2, 36)，这里的2就是u和v两个分量
# 绘制速度场的第一个分量

# 保存坐标数据
import numpy as np
import os

# 创建保存目录
save_dir = './saved_data'
os.makedirs(save_dir, exist_ok=True)

# 保存坐标数据
# coords_file = os.path.join(save_dir, 'coords.npy')
# np.save(coords_file, f.coords)
# print(f"Coordinate data saved to: {coords_file}")

# 保存速度数据
velocity_file = os.path.join(save_dir, 'velocity.npy')
np.save(velocity_file, f.u)
print(f"速度数据已保存至: {velocity_file}")

# 保存压力数据
pressure_file = os.path.join(save_dir, 'pressure.npy')
np.save(pressure_file, f.p)
print(f"压力数据已保存至: {pressure_file}")

# 保存元数据 (流场信息)
metadata = {
    'time': f.time,
    'nel': f.nelt,
    'ndims': f.ndims,
    'mesh_limits_x': [-15.0, 35.0],
    'mesh_limits_y': [-15.0, 15.0]
}
np.save(os.path.join(save_dir, 'metadata.npy'), metadata)
print(f"元数据已保存至: {os.path.join(save_dir, 'metadata.npy')}")

def calculate_drag_coefficient(coords, velocity, pressure, metadata):
    print("[DEBUG] Using calculate_drag_coefficient from readfiles.py")
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
    
    # 使用插值方法
    from scipy.interpolate import LinearNDInterpolator, griddata
    
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

# Load and visualize the saved data
def load_and_visualize(save_dir='./saved_data', output_prefix='flow_field'):
    """Load the saved data and generate visualization images"""
    # Load the data
    coords = np.load(os.path.join(save_dir, 'coords.npy'))
    velocity = np.load(os.path.join(save_dir, 'velocity.npy'))
    pressure = np.load(os.path.join(save_dir, 'pressure.npy'))
    metadata = np.load(os.path.join(save_dir, 'metadata.npy'), allow_pickle=True).item()
    
    print(f"Loaded data - Time step: {metadata['time']}, Number of elements: {metadata['nel']}")
    
    # Calculate the drag coefficient
    c_d = calculate_drag_coefficient(coords, velocity, pressure, metadata)
    print(f"Calculated drag coefficient: {c_d}")
    
    # Prepare the interpolation data
    x_coords = coords[:, 0, :].flatten()  # All x-coordinates
    y_coords = coords[:, 1, :].flatten()  # All y-coordinates
    u_component = velocity[:, 0, :].flatten()  # x-direction velocity
    v_component = velocity[:, 1, :].flatten()  # y-direction velocity
    
    # Create a regular grid
    x_min, x_max = metadata['mesh_limits_x']
    y_min, y_max = metadata['mesh_limits_y']
    grid_size = 500  # Grid resolution
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(xi, yi)
    
    # Calculate the grid spacing
    dx = (x_max - x_min) / (grid_size - 1)
    dy = (y_max - y_min) / (grid_size - 1)
    
    # Interpolate
    from scipy.interpolate import griddata
    
    # Velocity magnitude
    velocity_mag = np.sqrt(u_component**2 + v_component**2)
    u_grid = griddata((x_coords, y_coords), u_component, (X, Y), method='linear')
    v_grid = griddata((x_coords, y_coords), v_component, (X, Y), method='linear')
    vel_mag_grid = griddata((x_coords, y_coords), velocity_mag, (X, Y), method='linear')
    
    # Pressure
    p_grid = griddata((x_coords, y_coords), pressure.flatten(), (X, Y), method='linear')
    
    # Calculate the vorticity ω = ∂v/∂x - ∂u/∂y
    vorticity = np.zeros_like(u_grid)
    # Use central differences to calculate the derivatives
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            dv_dx = (v_grid[i, j+1] - v_grid[i, j-1]) / (2 * dx)
            du_dy = (u_grid[i+1, j] - u_grid[i-1, j]) / (2 * dy)
            vorticity[i, j] = dv_dx - du_dy
    
    # Plot the u-velocity component
    plt.figure(figsize=(16, 10))
    plt.contourf(X, Y, u_grid, levels=100, cmap='jet')
    plt.colorbar(label='$u$ (X-Velocity)')
    # Add a reference circle for the cylinder position
    circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=1, linewidth=1)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.title(f'$u$ Component (X-Velocity) at $t={metadata["time"]}$')
    plt.savefig(f'{output_prefix}_u_velocity.jpg', dpi=650)
    plt.close()
    
    # Plot the velocity magnitude
    plt.figure(figsize=(16, 10))
    plt.contourf(X, Y, vel_mag_grid, levels=100, cmap='viridis')
    plt.colorbar(label='$|\\vec{V}|$ (Velocity Magnitude)')
    # Add a reference circle for the cylinder position
    circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=1, linewidth=1)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.title(f'Velocity Magnitude at $t={metadata["time"]}$')
    plt.savefig(f'{output_prefix}_velocity_magnitude.jpg', dpi=650)
    plt.close()
    
    # Plot the pressure field
    plt.figure(figsize=(16, 10))
    plt.contourf(X, Y, p_grid, levels=100, cmap='coolwarm')
    plt.colorbar(label='$p$ (Pressure)')
    # Add a reference circle for the cylinder position
    circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=1, linewidth=1)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.title(f'Pressure Field at $t={metadata["time"]}$')
    plt.savefig(f'{output_prefix}_pressure.jpg', dpi=650)
    plt.close()
    
    # Plot the vorticity field
    plt.figure(figsize=(16, 10))
    # Use a red-blue contrast color map to highlight positive and negative vorticity
    plt.contourf(X, Y, vorticity, levels=100, cmap='RdBu_r', extend='both')
    plt.colorbar(label='$\\omega$ (Vorticity)')
    plt.axis('equal')
    plt.title(f'Vorticity Field at $t={metadata["time"]}$')
    # Add a reference circle for the cylinder position
    circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=1, linewidth=1)
    plt.gca().add_patch(circle)
    plt.savefig(f'{output_prefix}_vorticity.jpg', dpi=650)
    plt.close()
    
    # Plot the velocity vector field
    plt.figure(figsize=(16, 10))
    # Plot the velocity magnitude in the background
    plt.contourf(X, Y, vel_mag_grid, levels=50, cmap='viridis')
    # Downsample to avoid dense arrows
    skip = 20
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               u_grid[::skip, ::skip], v_grid[::skip, ::skip], 
               color='white', scale=25)
    plt.colorbar(label='$|\\vec{V}|$ (Velocity Magnitude)')
    # Add a reference circle for the cylinder position
    circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=1, linewidth=1)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.title(f'Velocity Vector Field at $t={metadata["time"]}$')
    plt.savefig(f'{output_prefix}_velocity_vectors.jpg', dpi=650)
    plt.close()
    
    # Vorticity and velocity vector overlay plot
    plt.figure(figsize=(16, 10))
    # Plot the vorticity in the background
    contour = plt.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r', extend='both')
    plt.colorbar(label='$\\omega$ (Vorticity)')
    # Overlay the velocity vectors
    skip = 25  # Use a larger interval for clarity
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               u_grid[::skip, ::skip], v_grid[::skip, ::skip], 
               color='black', scale=25, width=0.002)
    # Add a reference circle for the cylinder position
    circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=1, linewidth=1)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.title(f'Vorticity Field with Velocity Vectors at $t={metadata["time"]}$')
    plt.savefig(f'{output_prefix}_vorticity_vectors.jpg', dpi=650)
    plt.close()
    
    print(f"Visualization complete, images saved with prefix {output_prefix}")
    
    return c_d  # Return the drag coefficient for further use

# Process a single file
# load_and_visualize()

# Process all flow field files
def process_all_field_files(base_pattern='./ext_cyl0.f*', save_dir='./saved_data', debug=True):
    """
    Process all time step files and calculate the drag coefficient
    """
    import glob
    import re
    import matplotlib.pyplot as plt
    from fld_data import FldData
    
    # Get all matching files
    field_files = glob.glob(base_pattern)
    
    # Sort the files
    def extract_number(filename):
        # Extract the number from the filename
        match = re.search(r'f(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    field_files.sort(key=extract_number)
    
    if debug:
        print(f"Found {len(field_files)} flow field files:")
        for f in field_files:
            print(f"  - {f}")
    else:
        print(f"Found {len(field_files)} flow field files")
    
    # Create the storage directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the result lists
    times = []
    drag_coefficients = []
    
    # Iterate over each file
    for i, file_path in enumerate(field_files):
        # Output the completion percentage to reduce redundant logs
        if i % 10 == 0 or i == len(field_files)-1:
            print(f"\nProcessing file {i+1}/{len(field_files)} ({(i+1)/len(field_files)*100:.1f}%): {file_path}")
        else:
            print(f"Processing: {i+1}/{len(field_files)} ({(i+1)/len(field_files)*100:.1f}%)...", end="\r")
        
        try:
            # Load the data
            f = FldData.fromfile(file_path)
            
            print(f"Time step: {f.time}, Number of elements: {f.nelt}")
            
            # Process the coordinate data - only check and save at the first file, use the saved coordinates for subsequent files
            if i == 0:
                if f.coords is None or len(f.coords) == 0:
                    print(f"Error: First file {file_path} has no coordinate data, cannot continue processing")
                    return
                coords = f.coords
                # Save the coordinate data for use in subsequent files
                coords_file = os.path.join(save_dir, 'coords.npy')
                np.save(coords_file, coords)
                print(f"Coordinate data saved to: {coords_file}")
            else:
                # For subsequent files, use the saved coordinate data
                coords = np.load(os.path.join(save_dir, 'coords.npy'))
            
            # Save the data
            if i == 0:  # Only need to save the coordinates once
                coords_file = os.path.join(save_dir, 'coords.npy')
                np.save(coords_file, f.coords)
                print(f"Coordinate data saved to: {coords_file}")
            
            # Create a subdirectory for the current time step
            time_dir = os.path.join(save_dir, f"t_{f.time:.6f}")
            os.makedirs(time_dir, exist_ok=True)
            
            # Save the velocity data
            velocity_file = os.path.join(time_dir, 'velocity.npy')
            np.save(velocity_file, f.u)
            
            # Save the pressure data
            pressure_file = os.path.join(time_dir, 'pressure.npy')
            np.save(pressure_file, f.p)
            
            # Save the metadata
            metadata = {
                'time': f.time,
                'nel': f.nelt,
                'ndims': f.ndims,
                'mesh_limits_x': [-15.0, 35.0],
                'mesh_limits_y': [-15.0, 15.0]
            }
            np.save(os.path.join(time_dir, 'metadata.npy'), metadata)
            
            # Calculate the drag coefficient
            c_d = calculate_drag_coefficient(coords, f.u, f.p, metadata)
            print(f"Drag coefficient at time {f.time}: {c_d}")
            
            # Add to the result lists
            times.append(f.time)
            drag_coefficients.append(c_d)
            
            # Only generate visualizations for a few time steps to reduce computation
            # Generate visualizations for the first, last, and every 50th time step
            if i == 0 or i == len(field_files)-1 or (i > 0 and i % 50 == 0):
                # Generate the visualization
                output_prefix = os.path.join(time_dir, f"flow_field_t{f.time:.1f}")
                velocity = f.u
                pressure = f.p
                
                # Prepare the interpolation data
                x_coords = coords[:, 0, :].flatten()  # All x-coordinates
                y_coords = coords[:, 1, :].flatten()  # All y-coordinates
                u_component = velocity[:, 0, :].flatten()  # x-direction velocity
                v_component = velocity[:, 1, :].flatten()  # y-direction velocity
                
                # Create a regular grid - reduce the resolution to increase speed
                x_min, x_max = metadata['mesh_limits_x']
                y_min, y_max = metadata['mesh_limits_y']
                grid_size = 300  # Reduce the grid resolution to increase computation speed
                xi = np.linspace(x_min, x_max, grid_size)
                yi = np.linspace(y_min, y_max, grid_size)
                X, Y = np.meshgrid(xi, yi)
                
                # Calculate the grid spacing
                dx = (x_max - x_min) / (grid_size - 1)
                dy = (y_max - y_min) / (grid_size - 1)
                
                # Interpolate - use faster linear interpolation
                from scipy.interpolate import griddata
                
                # Only calculate the vorticity - directly interpolate without calculating other fields
                u_grid = griddata((x_coords, y_coords), u_component, (X, Y), method='linear')
                v_grid = griddata((x_coords, y_coords), v_component, (X, Y), method='linear')
                
                # Calculate the vorticity ω = ∂v/∂x - ∂u/∂y - use vectorized operations
                # Use vectorized operations instead of loops to calculate the vorticity
                vorticity = np.zeros_like(u_grid)
                
                # Create a central difference mask
                mask = np.ones_like(u_grid, dtype=bool)
                mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
                
                # Vectorized calculation of vorticity
                vorticity[1:-1, 1:-1] = (v_grid[1:-1, 2:] - v_grid[1:-1, :-2]) / (2 * dx) - \
                                       (u_grid[2:, 1:-1] - u_grid[:-2, 1:-1]) / (2 * dy)
                
                # 计算速度幅值用于绘图
                vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
                
                # 插值压力场
                p_grid = griddata((x_coords, y_coords), pressure.flatten(), (X, Y), method='linear')
                
                # 1. 绘制涡量场
                plt.figure(figsize=(12, 8))
                plt.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r', extend='both')
                plt.colorbar(label='$\\omega$ (Vorticity)')
                circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=0.7, linewidth=1)
                plt.gca().add_patch(circle)
                plt.axis('equal')
                plt.title(f'Vorticity Field at $t={metadata["time"]}$') 
                plt.savefig(f'{output_prefix}_vorticity.jpg', dpi=650)
                plt.close()
                
                # 2. 绘制速度幅值场
                plt.figure(figsize=(12, 8))
                plt.contourf(X, Y, vel_mag_grid, levels=50, cmap='viridis')
                plt.colorbar(label='$|\\vec{V}|$ (Velocity Magnitude)')
                circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=0.7, linewidth=1)
                plt.gca().add_patch(circle)
                plt.axis('equal')
                plt.title(f'Velocity Magnitude at $t={metadata["time"]}$')
                plt.savefig(f'{output_prefix}_velocity.jpg', dpi=650)
                plt.close()
                
                # 3. 绘制压力场
                plt.figure(figsize=(12, 8))
                plt.contourf(X, Y, p_grid, levels=50, cmap='coolwarm')
                plt.colorbar(label='$p$ (Pressure)')
                circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=0.7, linewidth=1)
                plt.gca().add_patch(circle)
                plt.axis('equal')
                plt.title(f'Pressure Field at $t={metadata["time"]}$')
                plt.savefig(f'{output_prefix}_pressure.jpg', dpi=650)
                plt.close()
                
                # 4. 速度矢量场与涡量场叠加
                plt.figure(figsize=(12, 8))
                plt.contourf(X, Y, vorticity, levels=30, cmap='RdBu_r', extend='both')
                plt.colorbar(label='$\\omega$ (Vorticity)')
                # 减少矢量箭头密度以提高可视化效果和速度
                skip = 35
                plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                           u_grid[::skip, ::skip], v_grid[::skip, ::skip], 
                           color='black', scale=30, width=0.002)
                circle = plt.Circle((0, 0), 0.5, color='gray', fill=True, alpha=0.7, linewidth=1)
                plt.gca().add_patch(circle)
                plt.axis('equal')
                plt.title(f'Vorticity Field with Velocity Vectors at $t={metadata["time"]}$')
                plt.savefig(f'{output_prefix}_vorticity_vectors.jpg', dpi=650)
                plt.close()
                
                print(f"Generated flow field visualizations for time step {f.time}")
        
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Plot the drag coefficient over time
    plt.figure(figsize=(12, 6))
    plt.plot(times, drag_coefficients, 'b-', linewidth=2)
    # plt.plot(times, drag_coefficients, 'ro', markersize=4)
    plt.grid(True)
    plt.xlabel('Time $t$')
    plt.ylabel('Drag Coefficient $C_D$')
    plt.title('Drag Coefficient vs Time')
    plt.savefig(os.path.join(save_dir, 'drag_coefficient_vs_time.jpg'), dpi=650)
    plt.close()
    
    # 保存阻力系数数据
    drag_data = np.column_stack((times, drag_coefficients))
    np.savetxt(os.path.join(save_dir, 'drag_coefficient_vs_time.txt'), 
               drag_data, 
               header='Time\tDrag_Coefficient', 
               delimiter='\t', 
               comments='')
    
    print("\nAll time steps processed")
    print(f"Drag coefficient data saved to: {os.path.join(save_dir, 'drag_coefficient_vs_time.txt')}")
    print(f"Drag coefficient vs time plot saved to: {os.path.join(save_dir, 'drag_coefficient_vs_time.jpg')}")

# 处理单个文件的情况
# load_and_visualize()

# 处理所有流场文件
process_all_field_files(base_pattern='../../YDW/NekExamples/ext_cyl/ext_cyl0.f*')


