# cost_surface_app_fixed.py  (put it next to your original file)
# ----------------------------------------------------------------
import numpy as np
import casadi as ca
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

from src.algorithms.classic.mpc_costs import COST_FUNCTION_MAP

state_labels = ["x", "theta", "x_dot", "theta_dot"]
state_indices = {lbl: i for i, lbl in enumerate(state_labels)}

# ----------------------------------------------------------------
# helper ----------------------------------------------------------
def evaluate_cost_surface(cost_key, var1, var2,
                          q_vals, r_val,
                          n_pts=120,
                          limits=None): # limits argument is unused, kept for signature stability
    
    # Base ranges (Updated with desired values)
    base_limits = {
        "x": (-1.0, 1.0),          # New base limit
        "theta": (-2 * np.pi, 2 * np.pi), # Keep fixed range
        "x_dot": (-5.0, 5.0),        # New base limit
        "theta_dot": (-10.0, 10.0),   # New base limit
    }

    # Get current weights
    w_x, w_theta, w_x_dot, w_theta_dot = q_vals
    weights = {
        "x": w_x,
        "theta": w_theta,
        "x_dot": w_x_dot,
        "theta_dot": w_theta_dot
    }

    # Determine dynamic limits
    v1_lims = list(base_limits[var1]) # Start with base
    v2_lims = list(base_limits[var2]) # Start with base

    # Use x_dot weight/range as reference for target cost
    target_max_cost = max(1.0, weights["x_dot"]) * (base_limits["x_dot"][1] ** 2)
    epsilon = 1e-6

    # Adjust var1 limits dynamically (if not theta)
    if var1 != 'theta':
        w1 = weights[var1]
        if w1 > epsilon:
            L1_desired = np.sqrt(target_max_cost / w1)
            L1 = min(L1_desired, base_limits[var1][1]) # Clamp to NEW base range magnitude
            v1_lims = [-L1, L1]
        # else: keep base_limits[var1]
            
    # Adjust var2 limits dynamically (if not theta)
    if var2 != 'theta':
        w2 = weights[var2]
        if w2 > epsilon:
            L2_desired = np.sqrt(target_max_cost / w2)
            L2 = min(L2_desired, base_limits[var2][1]) # Clamp to NEW base range magnitude
            v2_lims = [-L2, L2]
        # else: keep base_limits[var2]

    # Build square grid for the two chosen axes using determined limits
    v1_lo, v1_hi = v1_lims
    v2_lo, v2_hi = v2_lims
    # Ensure range isn't zero if limits are somehow equal
    if abs(v1_hi - v1_lo) < epsilon: v1_hi = v1_lo + 1.0
    if abs(v2_hi - v2_lo) < epsilon: v2_hi = v2_lo + 1.0
    
    x_grid = np.linspace(v1_lo, v1_hi, n_pts)
    y_grid = np.linspace(v2_lo, v2_hi, n_pts)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Prepare CasADi symbols and functions
    cost_class = COST_FUNCTION_MAP[cost_key]()
    Xk = ca.SX.sym("Xk", 4)
    Uk = ca.SX.sym("Uk", 1)         # symbolic control
    Q   = ca.diag(ca.vcat(q_vals))
    R   = ca.DM([[r_val]])
    Xrf = ca.DM.zeros(4, 1)
    stage_expr = cost_class.calculate_stage_cost(Xk, Uk, Xrf, Q, R)
    stage_expr = ca.substitute(stage_expr, Uk, ca.DM.zeros(1, 1))
    stage_fun = ca.Function("stage", [Xk], [stage_expr])

    # Evaluate on grid
    Z = np.zeros_like(X)
    for i in range(n_pts):
        for j in range(n_pts):
            s = np.zeros(4)
            s[state_indices[var1]] = X[i, j]
            s[state_indices[var2]] = Y[i, j]
            Z[i, j] = stage_fun(s).full().item()
            
    return X, Y, Z

# ----------------------------------------------------------------
# Dash layout -----------------------------------------------------
app = dash.Dash(__name__)
slider_marks = {0: "0", 50: "50", 100: "100"}

# Helper for sliders
def q_slider(slider_id, default):
    return dcc.Slider(
        id=slider_id, min=0, max=100, step=0.5, value=default,
        marks=slider_marks, tooltip={"placement": "bottom"},
        updatemode="drag",
    )

# --- Layout using Rows for compactness ---
app.layout = html.Div(
    [
        html.H3("MPC cost-surface explorer (cart-pole)"),

        # Row 1: Cost function and Variable Selectors
        html.Div([
            # Column 1.1: Cost Function
            html.Div([
                html.Label("Cost function"),
                dcc.Dropdown(
                    id="cost-key",
                    options=[{"label": k, "value": k} for k in COST_FUNCTION_MAP.keys()],
                    value="pendulum_swingup", clearable=False,
                ),
            ], style={'width': '28%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

            # Column 1.2: Var1 Selector
            html.Div([
                html.Label("Variable 1"),
                dcc.Dropdown(
                    id="var1", options=[{"label": s, "value": s} for s in state_labels],
                    value="theta", clearable=False,
                ),
            ], style={'width': '28%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),

            # Column 1.3: Var2 Selector
            html.Div([
                html.Label("Variable 2"),
                dcc.Dropdown(
                    id="var2", options=[{"label": s, "value": s} for s in state_labels],
                    value="theta_dot", clearable=False,
                ),
            ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top'}),
             # Column 1.4: Update Button (aligned right)
            html.Div([
                 html.Button("Update plot", id="update-btn", style={'marginTop': '25px'}) # Add margin for alignment
            ], style={'width': '10%', 'display': 'inline-block', 'textAlign': 'right', 'verticalAlign': 'bottom'}),

        ], style={'marginBottom': '10px'}), # End of Row 1

        # Row 2: Weight Sliders (arranged more compactly)
        html.Div([
            html.H5("Weights", style={'textAlign': 'center', 'marginBottom': '5px'}),
             # Sub-row for Q weights
             html.Div([
                 html.Div([html.Label("w_x"), q_slider("w-x", 1.0)], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                 html.Div([html.Label("w_theta"), q_slider("w-theta", 1.0)], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                 html.Div([html.Label("w_x_dot"), q_slider("w-xdot", 1.0)], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                 html.Div([html.Label("w_theta_dot"), q_slider("w-thdot", 1.0)], style={'width': '23%', 'display': 'inline-block'}),
             ], style={'marginBottom': '10px'}),
             # Sub-row for R weight (can be centered or left-aligned)
             html.Div([
                 html.Label("R (control weight)", style={'display': 'block'}), # Label above
                 q_slider("r-weight", 1.0)
             ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '25%'}), # Example centering

        ], style={'border': '1px solid lightgrey', 'padding': '10px', 'marginBottom': '20px'}), # End of Row 2

        # Row 3: The Plot
        dcc.Graph(id="cost-surface", style={"height": "80vh"}), # Restored height

    ], style={"fontFamily": "sans-serif", "margin": "18px"},
)

# ----------------------------------------------------------------
# callback --------------------------------------------------------
@app.callback(
    Output("cost-surface", "figure"),
    Input("update-btn", "n_clicks"),
    State("cost-key", "value"),
    State("var1", "value"),
    State("var2", "value"),
    State("w-x", "value"),
    State("w-theta", "value"),
    State("w-xdot", "value"),
    State("w-thdot", "value"),
    State("r-weight", "value"),
)
def update_plot(_, cost_key, var1, var2,
                w_x, w_theta, w_xdot, w_thdot, r_weight):

    if var1 == var2:
        return go.Figure(layout={"title": "Select two different state variables"})
    
    q_vec = np.array([w_x, w_theta, w_xdot, w_thdot])

    # Call evaluate_cost_surface (which calculates limits inside using updated BASE_LIMITS)
    X, Y, Z = evaluate_cost_surface(cost_key, var1, var2, q_vec, r_weight,
                                    n_pts=120)

    # Create Figure
    fig = go.Figure(
        data=[
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Plasma",
                contours={"z": {"show": True, "project": {"z": True}}},
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=var1,
            yaxis_title=var2,
            zaxis_title="stage cost",
            camera=dict(eye=dict(x=1.25, y=1.25, z=0.8)),
        ),
        title=f"{cost_key} cost – ({var1}, {var2}) plane",
    )
    return fig

# ----------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)