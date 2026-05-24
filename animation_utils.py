import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

def draw_or_gate(ax, x, y, w=1.5, h=1.0):
    """
    Draws a highly polished standard electronic OR gate symbol at (x, y).
    """
    # Bezier curves to model the iconic curved back and pointed tip of an OR gate
    verts = [
        (x - w/2, y - h/2),  # Bottom-left curve start
        (x - w/4, y),        # Curved back midpoint
        (x - w/2, y + h/2),  # Top-left curve end
        (x, y + h/2),        # Top flat/arc control point
        (x + w/2, y),        # Pointed tip
        (x, y - h/2),        # Bottom flat/arc control point
        (x - w/2, y - h/2),  # Close path
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='#c0392b', edgecolor='#ecf0f1', lw=1.5, zorder=3)
    ax.add_patch(patch)
    return patch

def create_3sat_circuit_animation(clause_idx, clause_sign, v_traj, time_traj, C_traj, batch_idx, output_path, fps=1,
                                  flash_decay=0.1, flip_stability=1, flip_magnitude_threshold=0.0):
    """
    Generates an optimized high-quality video/GIF of the 3SAT solver circuit representation for all timesteps,
    visualizing the spatiotemporal avalanches of spin flips in real time.

    Avalanche visibility parameters
    --------------------------------
    flash_decay : float
        Multiplicative decay applied to the glow each frame (0 < flash_decay < 1).
        Lower values fade faster. Default 0.5 → glow gone in ~5 frames.
    flip_stability : int
        Number of consecutive frames the new sign must persist before a flash is
        triggered. Filters out rapid back-and-forth oscillations. Default 2.
    flip_magnitude_threshold : float
        Minimum |v| required at the *new* sign after a crossing for a flash to
        fire. Ignores near-zero jitter crossings (e.g. v going -0.01 → +0.01).
        Default 0.05.
    """
    n_clause, n_literals = clause_idx.shape  # E.g., (43, 3)
    
    # 1. Compute a clean grid layout for the OR gates
    cols = int(np.ceil(np.sqrt(n_clause)))
    rows = int(np.ceil(n_clause / cols))
    
    # 2. Setup the Matplotlib figure with a sleek dark-mode background and low DPI for fast rendering
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#111111', dpi=90)
    ax.set_facecolor('#111111')
    ax.set_xlim(-2, cols * 4)
    ax.set_ylim(-2, rows * 4)
    ax.axis('off')
    
    # Pre-calculate cell positions
    gate_centers = []
    for idx in range(n_clause):
        r = idx // cols
        c = idx % cols
        gate_centers.append((c * 4, r * 4))

    # Pre-draw static background elements and cache references to dynamic artists
    wire_lines = []
    gate_patches = []
    
    # Create the custom Red-White-Green colormap for clause satisfaction (1 - C)
    cmap_gate = LinearSegmentedColormap.from_list('gate_satisfaction', ['#c0392b', '#ffffff', '#27ae60'])
    flash_rgb = np.array([0.95, 0.77, 0.06, 1])  # Vibrant Amber/Yellow (#f1c40f)
    
    # Keep track of previous spins and current flash intensities for avalanche visualization
    n_var = v_traj[0].shape[-1]
    v_prev = v_traj[0][batch_idx].cpu().numpy()
    flash_intensities = np.zeros(n_var)

    # Stability gate: track how many consecutive frames each variable has held its
    # current (candidate) sign since the last crossing.
    # Initialise so every variable is considered stable from the start.
    candidate_sign = np.sign(v_prev)          # the sign we are waiting to confirm
    stability_counter = np.full(n_var, flip_stability, dtype=int)  # already stable
    
    for idx, (x, y) in enumerate(gate_centers):
        # Draw OR gate shape
        gate_patch = draw_or_gate(ax, x, y)
        gate_patches.append(gate_patch)
        
        # Add a clause label (Static Background Artist)
        ax.text(x, y + 0.8, f"C_{idx}", color='#95a5a6', fontsize=8, ha='center', fontweight='bold')
        
        y_offsets = [0.25, 0.0, -0.25]
        clause_wires = []
        for k in range(3):
            # Input wire coordinates
            wire_x = [x - 1.5, x - 0.5]
            wire_y = [y + y_offsets[k], y + y_offsets[k]]
            
            # Create a thick line for the wire (Dynamic Foreground Artist)
            line, = ax.plot(wire_x, wire_y, lw=8, color='gray', solid_capstyle='round', zorder=2)
            clause_wires.append(line)
            
            # Label near the wire start: e.g. "v_3" or "¬v_3" (Static Background Artist)
            var_num = clause_idx[idx, k].item()
            sign = clause_sign[idx, k].item()
            label_text = f"v{var_num}" if sign > 0 else f"¬v{var_num}"
            
            ax.text(x - 1.6, y + y_offsets[k], label_text, color='#bdc3c7', fontsize=7, ha='right', va='center')
            
        wire_lines.append(clause_wires)
        
    title_text = ax.text(cols * 2 - 1, rows * 4 - 0.5, "", color='#ffffff', fontsize=14, ha='center', fontweight='bold')
    
    # Total frames is the actual number of simulation steps saved in trajectories
    total_frames = len(v_traj)
    
    def update(frame_idx):
        nonlocal v_prev, flash_intensities, candidate_sign, stability_counter
        v_vals = v_traj[frame_idx][batch_idx].cpu().numpy()
        t_val = time_traj[batch_idx][frame_idx].item()
        
        # C_vals represents clause states at this step for the batch
        C_vals = C_traj[frame_idx][batch_idx]  # Shape: (n_clause,)
        
        # 1. Detect raw sign crossings vs the last committed sign
        current_sign = np.sign(v_vals)  # -1, 0, or +1 per variable
        crossed = current_sign != candidate_sign  # started moving to a new sign

        flash_fired = True

        # 2. Update stability counters and decide which crossings are committed
        for p in range(n_var):
            if crossed[p]:
                # Crossing detected: reset counter and update candidate
                candidate_sign[p] = current_sign[p]
                stability_counter[p] = 1
                flash_fired = False
            else:
                # Still on the same side: increment (cap at flip_stability)
                stability_counter[p] = min(stability_counter[p] + 1, flip_stability)

            # A flip is "committed" when:
            #   (a) the variable just reached the required stability duration, AND
            #   (b) |v| at the new position is above the magnitude threshold (not jitter)
            just_committed = (stability_counter[p] == flip_stability) and not flash_fired
            above_threshold = abs(v_vals[p]) >= flip_magnitude_threshold

            if just_committed and above_threshold:
                # Only flash if the sign actually changed from the initially stable sign
                # (stability_counter hits flip_stability on init too, so guard with frame > 0)
                if frame_idx >= flip_stability:
                    flash_intensities[p] = 1.0
                    flash_fired = True

            else:
                flash_intensities[p] *= flash_decay
                
        # Cache current voltages as previous for the next frame
        v_prev = v_vals.copy()
        
        # 3. Update wire and gate rendering
        for idx in range(n_clause):
            # Update input wire colors and dynamic thickness
            for k in range(3):
                var_num = clause_idx[idx, k].item()
                sign = clause_sign[idx, k].item()
                # Correct by sign and map to [0, 1]
                satisfaction = (v_vals[var_num] * sign + 1.0) / 2.0
                # satisfaction = v_vals[var_num] * sign
                
                # Dynamic wire color: blend satisfaction grayscale with radiant amber based on flash intensity
                # gray_val = 0.15 + 0.85 * satisfaction
                base_color = np.array(cmap_gate(satisfaction))
                # base_color = np.array([wire_color, wire_color, wire_color])
                
                F_p = flash_intensities[var_num]
                blended_color = (1.0 - F_p) * base_color + F_p * flash_rgb
                
                wire_lines[idx][k].set_color(blended_color)
                # Swell line width dynamically on flip pulse
                wire_lines[idx][k].set_linewidth(8.0 + 6.0 * F_p)
            
            # Update gate fill color using the 1 - C metric mapped to Red-White-Green
            c_val = C_vals[idx].item()
            satisfaction_metric = np.clip(1.0 - c_val, 0.0, 1.0)
            gate_color = cmap_gate(satisfaction_metric)
            gate_patches[idx].set_facecolor(gate_color)
            
        # Update header title text
        title_text.set_text(f"DMM 3SAT Solver | Step {frame_idx} | Time: {t_val:.2f}s")
        
        # Return only the modified dynamic artists for blitting optimization
        flat_lines = [line for sublist in wire_lines for line in sublist]
        return flat_lines + gate_patches + [title_text]

    # Assemble the animation with blitting enabled
    ani = FuncAnimation(fig, update, frames=total_frames, blit=True)
    
    # Save the animation (MP4 if ffmpeg is found, otherwise fallback to GIF via Pillow)
    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps, bitrate=1500, codec='h264')
        ani.save(f"{output_path}.mp4", writer=writer)
        print(f"Animation saved successfully as MP4 to {output_path}.mp4")
    except Exception as e:
        # Fallback to GIF if ffmpeg is missing
        print(f"FFmpeg not available ({e}). Fallback: saving as optimized GIF...")
        ani.save(f"{output_path}.gif", writer='pillow', fps=fps, dpi=90, savefig_kwargs={'facecolor':'#111111'})
        print(f"Animation saved as GIF to {output_path}.gif")
        
    plt.close(fig)
