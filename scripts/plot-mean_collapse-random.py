import numpy as np
import plotly.graph_objects as go

# Set a seed for reproducibility
np.random.seed(42)

def generate_circle(
    n_points: int,
    center: tuple[float, float],
    radius: float,
    noise: float = 0.15
) -> np.ndarray:
    """
    Generates points in a ring shape around a center.

    Parameters
    ----------
    n_points : int
        The number of points to generate.
    center : tuple[float, float]
        The (x, y) coordinates of the circle's center.
    radius : float
        The radius of the circle.
    noise : float
        The standard deviation of the noise added to the radius.

    Returns
    -------
    np.ndarray
        An array of shape (n_points, 2) containing the generated points.
    """
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = radius + noise * np.random.randn(n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))

# --- 1. Define the data points as rings using the new parameters ---

# A ring of source (control) cells
control_points = generate_circle(n_points=100, center=(0, 0), radius=1.0, noise=0.15)

# A ring of target (perturbed) cells under a specific condition
perturbed_points = generate_circle(n_points=150, center=(3, 3), radius=1.5, noise=0.2)

# Calculate the actual mean of each noisy distribution
control_mean = np.mean(control_points, axis=0)
perturbed_mean = np.mean(perturbed_points, axis=0)

# --- 2. Create the Plotly figure ---

fig = go.Figure()

# --- 3. Plot the ambiguous random pairing problem ---

# Draw faint gray lines from a few source points to all possible targets
representative_sources = [50]
for source_idx in representative_sources:
    for j in range(perturbed_points.shape[0]):
        fig.add_trace(go.Scatter(
            x=[control_points[source_idx, 0], perturbed_points[j, 0]],
            y=[control_points[source_idx, 1], perturbed_points[j, 1]],
            mode='lines',
            line=dict(
                color='gray',
                width=1.0,
                # dash='dot'
            ),
            showlegend=False,
            hoverinfo='none'
        ))

# Plot the target points
fig.add_trace(go.Scatter(
    x=perturbed_points[:, 0],
    y=perturbed_points[:, 1],
    mode='markers',
    marker=dict(color='royalblue', size=8, opacity=0.8, line=dict(width=1, color='White')),
    name='Perturbed Cells',
    showlegend=False
))

# Plot the source points
fig.add_trace(go.Scatter(
    x=control_points[:, 0],
    y=control_points[:, 1],
    mode='markers',
    marker=dict(color='crimson', size=9, opacity=0.8, line=dict(width=1, color='White')),
    name='Control Cells',
    showlegend=False
))


# --- 4. Plot the trivial mean collapse solution ---

# Plot the mean of the control distribution
# fig.add_trace(go.Scatter(
#     x=[control_mean[0]],
#     y=[control_mean[1]],
#     mode='markers',
#     marker=dict(color='crimson', size=18,
#                 # symbol='x-thin',
#                 line=dict(width=3, color='black')),
#     name='Mean of Controls',
#     showlegend=False
# ))

#? Plot the mean of the perturbed distribution
fig.add_trace(go.Scatter(
    x=[perturbed_mean[0]],
    y=[perturbed_mean[1]],
    mode='markers',
    marker=dict(
        color='royalblue',
        size=18,
        # symbol='cross-thin',
        symbol='star',
        line=dict(width=3, color='black')),
    name='Mean of Perturbed',
    showlegend=False
))

# --- 5. Finalize plot styling ---

#? --- Calculate plot range for axis arrows ---
all_points = np.vstack([control_points, perturbed_points])
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
x_span = (x_max - x_min) * 1.25
y_span = (y_max - y_min) * 1.25
max_span = max(x_span, y_span)
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
x_range = [x_center - max_span / 2, x_center + max_span / 2]
y_range = [y_center - max_span / 2, y_center + max_span / 2]


fig.update_layout(
    # title=dict(text='<b>How Random Pairing Leads to Mean Collapse</b>', font=dict(size=20), x=0.5),
    template='plotly_dark',
    width=1000,
    height=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title=None,
        range=x_range
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title=None,
        scaleanchor="x",
        scaleratio=1,
        range=y_range
    ),
    annotations=[
        # --- The main "Mean Collapse" arrow ---
        go.layout.Annotation(
            # x=perturbed_mean[0],
            # y=perturbed_mean[1],
            # ax=control_mean[0],
            # ay=control_mean[1],
            ax=control_points[source_idx, 0],
            ay=control_points[source_idx, 1],
            x=perturbed_mean[0],
            y=perturbed_mean[1],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor='limegreen'
        ),
        # --- Annotation for the mean collapse concept ---
        # go.layout.Annotation(
        #     x=perturbed_mean[0] + 0.5, y=perturbed_mean[1] + 2.0,
        #     text="Model learns to map the<br>average control state to the<br>average perturbed state",
        #     showarrow=True, arrowhead=4,
        #     ax=perturbed_mean[0] + 0.1, ay=perturbed_mean[1] + 0.1,
        #     bgcolor="rgba(255, 255, 0, 0.7)", bordercolor="black", borderwidth=1,
        # ),
        # # --- The "Perturbation Label" text box ---
        # go.layout.Annotation(
        #     x=0.97, y=0.03, xref='paper', yref='paper',
        #     text='<b>Perturbation Label:</b><br>Gene A Knockout',
        #     showarrow=False, align='right', font=dict(size=12),
        #     bgcolor="rgba(200, 230, 255, 0.8)", bordercolor="royalblue", borderwidth=1, borderpad=8
        # ),
        # # --- Add a text label for the green arrow ---
        # go.layout.Annotation(
        #     x=control_mean[0] + (perturbed_mean[0] - control_mean[0])/2,
        #     y=control_mean[1] + (perturbed_mean[1] - control_mean[1])/2 + 0.3,
        #     text="Trivial Solution<br>(Mean Collapse)",
        #     showarrow=False, font=dict(color='limegreen', size=12)
        # ),
        # # --- X-axis arrow ---
        # go.layout.Annotation(
        #     x=x_range[1], y=y_range[0], ax=x_range[0], ay=y_range[0],
        #     xref='x', yref='y', axref='x', ayref='y',
        #     showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.5, arrowcolor='black'
        # ),
        # # --- Y-axis arrow ---
        # go.layout.Annotation(
        #     x=x_range[0], y=y_range[1], ax=x_range[0], ay=y_range[0],
        #     xref='x', yref='y', axref='x', ayref='y',
        #     showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1.5, arrowcolor='black'
        # )
    ]
)

# --- 6. Save and show the plot ---
# fig.write_html('random_pairing_causes_mean_collapse.html')
# fig.show()
png_path = 'plot-mean_collapse-random.png'
fig.write_image(png_path, scale=2)   # `scale` ↑ → higher‑resolution PNG

