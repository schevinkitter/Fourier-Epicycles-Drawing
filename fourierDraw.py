import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import animation
from matplotlib.lines import Line2D
import scienceplots
import os

plt.style.use('science')
plt.style.use('dark_background')


def get_dft(coords):
    """
    Return:
      coeffs: FFT coefficients normalized such that reconstruction is sum(c * exp(2j*pi*freq*t))
      amps: absolute values of coeffs
      phases: angles
      freqs: frequency indices (cycles per period)
    """
    N = len(coords)
    # Use norm='forward' so coefficients are X_k / N and reconstruction is sum(c * exp(2j*pi*k*t))
    coeffs = np.fft.fft(coords, norm='forward')
    amps = np.abs(coeffs)
    phases = np.angle(coeffs)
    freqs = np.fft.fftfreq(N, d=1.0/N)  # returns integer-like frequencies: 0,1,2,...,-N/2,...
    return coeffs, amps, phases, freqs


def pos_at_time(t, coeffs, freqs):
    """
    t: normalized time in [0,1)
    coeffs: complex coefficients (already filtered & ordered)
    freqs: corresponding frequencies (same length as coeffs)
    returns list of centers (x,y) for each circle, starting with origin (0,0) then each subsequent center.
    """
    loc = 0 + 0j
    centers = [(0.0, 0.0)]
    # use complex addition of each vector
    for c, f in zip(coeffs, freqs):
        vec = c * np.exp(2j * np.pi * f * t)
        loc = loc + vec
        centers.append((loc.real, loc.imag))
    return centers


def get_path_at_time(t, coeffs, freqs):
    
    """
    t: normalized time in [0,1)
    coeffs: complex coefficients (already filtered & ordered)
    freqs: corresponding frequencies (same length as coeffs)
    returns the tip position (x,y) at time t.
    """
    loc = 0 + 0j
    # use complex addition of each vector
    for c, f in zip(coeffs, freqs):
        vec = c * np.exp(2j * np.pi * f * t)
        loc = loc + vec
    return (loc.real, loc.imag) 

    
def get_init_vals_from_coeffs(coeffs, freqs):
    """
    Create Circle patches and Arrow patches from filtered coeffs & freqs.
    Radii = abs(coeffs). The arrays are expected to be in the chain order:
    base (big) -> ... -> tip (small)
    """
    circles = []
    arrows = []
    init_centers = pos_at_time(0.0, coeffs, freqs)  # centers at t=0

    # create circle patches (skip the origin entry at index 0 but keep radius for each coefficient)
    for i, center in enumerate(init_centers[1:], start=0):  # i corresponds to coeff index
        radius = np.abs(coeffs[i])
        alpha = 0.2
        circ = Circle(center, radius, fill=False, alpha=alpha, color='white')
        circles.append(circ)

    # create arrows connecting centers (len(arrows) == len(circles))
    for i in range(len(init_centers) - 1):
        start_pos = init_centers[i]
        end_pos = init_centers[i + 1]
        arrow = FancyArrowPatch(posA=start_pos, posB=end_pos, arrowstyle='-|>',
                                mutation_scale=5, color='#FFA500', alpha=0.6, shrinkA=0, shrinkB=0)
        arrows.append(arrow)

    return circles, arrows


def main():
    # load data
    data = create_complex_array_from_csv("example.csv")
    # data = order_points_nearest_neighbor(data) # order the points if not already ordered!
    data = data[::1] # resample if needed
    
    # Scale the data so the drawing fits nicely on the standard plot axes.
    # Find the maximum dimension (radius)
    max_dim = np.max(np.abs(data))
    if max_dim > 1.0: # Only scale if the dimensions are large
        # Scale to fit roughly within a radius of 2.0 for visibility
        data = data * (3.0 / max_dim)
        print(f"Data scaled by factor {2.0 / max_dim:.4f} to fit visualization area.")
    x = data.real
    y = data.imag


    N_orig = len(data)
    coeffs, amps, phases, freqs = get_dft(data)

    # Filter by amplitude (keep components above a threshold fraction of max amplitude)
    amp_threshold = 0 * np.max(amps)  # adjust (0.001, 0.0001, etc.)
    keep_mask = amps >= amp_threshold

    coeffs_f = coeffs[keep_mask]
    amps_f = amps[keep_mask]
    freqs_f = freqs[keep_mask]

    print(f"Kept {len(coeffs_f)} / {N_orig} coefficients after amplitude filtering")

    # Sort by amplitude descending so large (low-frequency / important) circles are first
    order = np.argsort(amps_f)[::-1]
    coeffs_f = coeffs_f[order]
    amps_f = amps_f[order]
    freqs_f = freqs_f[order]

    # Now reverse so that the chain goes from large -> ... -> small (small last, near tip)
    # after pos_at_time we add vectors sequentially; the last added vector will be the tip's last small vector
    coeffs_chain = coeffs_f
    freqs_chain = freqs_f
    amps_chain = amps_f
    
    t = np.linspace(0, 1, 5000, endpoint=False)
    pathX = np.zeros_like(t)
    pathY = np.zeros_like(t)
    for i, ti in enumerate(t):
        tip_pos = get_path_at_time(ti, coeffs_chain, freqs_chain)
        # You can store or use tip_pos as needed for further analysis or visualization  
        pathX[i] = tip_pos[0]
        pathY[i] = tip_pos[1]   
    
    # visualize
    plt.figure()
    plt.scatter(x, y, s=10, color='gray', alpha=0.3, label='original')
    plt.scatter(pathX, pathY, s=10, color='cyan', label='cropped')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

    # Build visualization objects
    circles, arrows = get_init_vals_from_coeffs(coeffs_chain, freqs_chain)

    # Drop the first circle/arrow if you'd like to hide DC's circle (optional)
    # But be careful: DC term shifts the entire drawing, so hiding it may make positions misleading.
    # If you want to keep the drawing centered, do NOT remove the DC term from the chain.
    # Here we keep everything (no pop).

    # Prepare figure
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax = plt.axes()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)


    # Add circle and arrow patches
    for circ in circles:
        ax.add_patch(circ)
    for arrow in arrows:
        ax.add_patch(arrow)

    # path
    path_x, path_y = [], []
    drawing_path_color = '#00FFFF'
    drawer_point_color = '#FFFF00'
    path_line = Line2D([], [], color=drawing_path_color, lw=3)
    ax.add_line(path_line)
    dot_final = ax.scatter([], [], lw=1, color=drawer_point_color)

    total_frames = N_orig  # animate over original sample count
    def init_anim():
        return circles + arrows + [path_line, dot_final]

    def animate(frame_idx):
        # convert frame index to normalized time t in [0,1)
        t = (frame_idx % total_frames) / total_frames

        centers = pos_at_time(t, coeffs_chain, freqs_chain)

        # update circles (circle i corresponds to centers[i+1], radius = abs(coeffs_chain[i]))
        for i, circ in enumerate(circles):
            circ.center = centers[i + 1]  # center of circle i

        # update arrows
        for i, arr in enumerate(arrows):
            start = centers[i]
            end = centers[i + 1]
            arr.set_positions(start, end)

        # update path tip (last center)
        tip = centers[-1]
        path_x.append(tip[0])
        path_y.append(tip[1])
        path_line.set_data(path_x, path_y)

        # update final dot
        dot_final.set_offsets([tip])

        return circles + arrows + [path_line, dot_final]

    anim = animation.FuncAnimation(fig, animate, init_func=init_anim,
                                   frames=len(amps_chain) - 1, interval=25, blit=False, repeat=False)

    # plt.show()
    anim.save('example.gif', fps=30)  # optional save


def create_complex_array_from_csv(csv_path):
    """
    Loads a 2D array of (X, Y) coordinates from a CSV, transforms it
    into a 1D flat array of complex numbers (X + iY), and plots the coordinates.
    
    Returns: A flat NumPy array of complex numbers (X + iY) or None if file error.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        print("Please ensure you have run the web tool and saved the CSV file.")
        return None

    try:
        # 1. Load the 2D data array (N rows, 2 columns: [X, Y])
        data_2d = np.loadtxt(
            csv_path, 
            delimiter=',', 
            skiprows=1, 
            dtype=float 
        )
        
        print(f"Loaded 2D array of coordinates. Shape: {data_2d.shape}")
        
        # 2. Separate the X and Y components (slicing the columns)
        x_coords = data_2d[:, 0] # All rows, column 0 (Real part)
        y_coords = data_2d[:, 1] # All rows, column 1 (Imaginary part)
        
        # 3. Vectorized Complex Number Construction
        # This is the line that performs the desired operation: X + iY
        complex_array_flat = x_coords + 1j * y_coords
        
        print("-" * 40)
        print(f"Complex Array created successfully.")
        print(f"Final Shape: {complex_array_flat.shape}")
        print("\nFirst 5 Complex Numbers (X + iY):")
        print(complex_array_flat[:5])
        
        
        # print("\nGenerating coordinate plot for verification...")
        # plt.figure(figsize=(8, 8))
        # # Scatter plot: use small marker size (s=1) for dense coordinate data
        # plt.scatter(x_coords, y_coords, s=1, color='darkgreen') 
        
        # plt.title('Visualization of Logo Edge Coordinates')
        # plt.xlabel('X Coordinate (Real Part)')
        # plt.ylabel('Y Coordinate (Imaginary Part)')
        
        # # Set aspect ratio to equal to correctly display shapes
        # plt.gca().set_aspect('equal', adjustable='box') 
        # # Invert Y axis for typical image coordinate system (origin top-left)
        # plt.gca().invert_yaxis()
        
        # plt.show()
        return complex_array_flat

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        print("Please verify the CSV file is correctly formatted (two integer/float columns).")
        return None
    
 
def order_points_nearest_neighbor(points):
    n = len(points)
    ordered = [points[0]]
    remaining = list(points[1:])
    for _ in range(n - 1):
        last = ordered[-1]
        dists = np.abs(np.array(remaining) - last)
        nearest_idx = np.argmin(dists)
        ordered.append(remaining.pop(nearest_idx))
    return np.array(ordered)
        
        
if __name__ == "__main__":
    main()
