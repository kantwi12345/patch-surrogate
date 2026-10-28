import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.invert import design_from_spec
from src.sim.schema import DesignParams
from src.sim.analytical_model import simulate

try:
    from src.sim.hfss_model import simulate_hfss, simulate_hfss_with_pattern
    HFSS_AVAILABLE = True
except Exception:
    HFSS_AVAILABLE = False

st.set_page_config(
    page_title="Patch Antenna Designer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.85; font-size: 1.05rem; }
    .result-card {
        background: #f7f9fb;
        border: 1px solid #e3e8ee;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    .status-pass { color: #1a7f37; font-weight: 600; font-size: 1.1rem; }
    .status-fail { color: #cf222e; font-weight: 600; font-size: 1.1rem; }
    .accuracy-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e3e8ee;
        border-radius: 10px;
        padding: 0.8rem;
        min-height: 90px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"] {
        color: #5a6472 !important;
        white-space: normal !important;
        overflow: visible !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #111 !important;
        font-size: 1.3rem !important;
        white-space: normal !important;
        overflow: visible !important;
    }
    div[data-testid="stMetricDelta"] { color: #444 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📡 Patch Antenna Inverse Design</h1>
    <p>CAD-style workspace — tune geometry live, inspect physics, validate against real HFSS.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HFSS calibration data (from M5_RESULTS.md, 8 real full-wave validation runs)
# ---------------------------------------------------------------------------
CALIBRATION_FREQ_GHZ = np.array([1.2, 1.8, 2.4, 3.0, 3.5, 4.2, 5.0, 5.8])
CALIBRATION_FREQ_ERR_PCT = np.array([4.1, 1.5, 0.9, 0.7, 2.7, 6.7, 11.9, 3.4])
CALIB_MIN, CALIB_MAX = CALIBRATION_FREQ_GHZ.min(), CALIBRATION_FREQ_GHZ.max()


def estimate_accuracy(freq_ghz):
    extrapolated = freq_ghz < CALIB_MIN or freq_ghz > CALIB_MAX
    err_pct = float(np.interp(freq_ghz, CALIBRATION_FREQ_GHZ, CALIBRATION_FREQ_ERR_PCT))
    return err_pct, extrapolated


def accuracy_badge_html(err_pct, extrapolated):
    if err_pct < 2:
        color, label = "#1a7f37", "HIGH"
    elif err_pct < 7:
        color, label = "#9a7b00", "MODERATE"
    else:
        color, label = "#cf222e", "LOW"
    extra = " (extrapolated — outside calibrated 1.2-5.8GHz range)" if extrapolated else ""
    return (
        f'<span class="accuracy-badge" style="background:{color}20;color:{color};">'
        f'{label} confidence — expected freq error ≈ ±{err_pct:.1f}%{extra}</span>'
    )


def auto_relax(freq_ghz, min_s11_db, gain_dbi, max_steps=12, s11_step=2, gain_step=0.5,
                s11_floor=-5, gain_floor=0.0):
    """
    Try the requested S11 (and, if set, gain) target together. If either
    isn't met, progressively relax whichever constraint(s) failed and
    retry -- S11 toward -5dB (least strict allowed), gain toward 0dBi --
    up to max_steps times. Returns the best result found, the final
    (possibly relaxed) targets, and a full attempt log for transparency.
    """
    attempts = []
    target_s11 = min_s11_db
    target_gain = gain_dbi
    result, gt = None, None

    for _ in range(max_steps):
        result = design_from_spec(
            target_freq_hz=freq_ghz * 1e9,
            target_min_s11_db=target_s11,
            target_gain_dbi=target_gain,
        )
        gt = result["ground_truth"]
        s11_met = gt.min_s11_db <= target_s11
        gain_met = (target_gain is None) or (gt.peak_gain_dbi >= target_gain)

        attempts.append({
            "target_s11_db": target_s11, "achieved_s11_db": gt.min_s11_db,
            "target_gain_dbi": target_gain,
            "achieved_gain_dbi": gt.peak_gain_dbi,
            "met": s11_met and gain_met,
        })

        if s11_met and gain_met:
            return result, gt, target_s11, target_gain, attempts

        new_s11, new_gain = target_s11, target_gain
        if not gain_met and target_gain is not None:
            new_gain = max(target_gain - gain_step, gain_floor)
        if not s11_met:
            new_s11 = min(target_s11 + s11_step, s11_floor)

        if (new_s11, new_gain) == (target_s11, target_gain):
            break
        target_s11, target_gain = new_s11, new_gain

    return result, gt, target_s11, target_gain, attempts


def compute_vswr(s11_db):
    gamma = np.clip(10 ** (np.asarray(s11_db) / 20.0), 0, 0.999)
    return (1 + gamma) / (1 - gamma)


# ---------------------------------------------------------------------------
# 3D CAD-style geometry schematic (inset-fed microstrip patch)
# ---------------------------------------------------------------------------
def box_mesh(xr, yr, zr, color, opacity=1.0, name="", showlegend=True):
    x0, x1 = xr
    y0, y1 = yr
    z0, z1 = zr
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name, flatshading=True, showlegend=showlegend,
        lighting=dict(ambient=0.85, diffuse=0.4, specular=0.05, roughness=0.9, fresnel=0.05),
        lightposition=dict(x=100, y=200, z=300),
    )


def box_edges(xr, yr, zr, color="black", width=3):
    x0, x1 = xr
    y0, y1 = yr
    z0, z1 = zr
    corners = {
        0: (x0, y0, z0), 1: (x1, y0, z0), 2: (x1, y1, z0), 3: (x0, y1, z0),
        4: (x0, y0, z1), 5: (x1, y0, z1), 6: (x1, y1, z1), 7: (x0, y1, z1),
    }
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                         line=dict(color=color, width=width),
                         showlegend=False, hoverinfo="skip")


def build_3d_view(L_mm, W_mm, h_mm, y0_mm):
    gnd_margin, sub_margin = 8.0, 3.0
    gnd_thk, patch_thk, line_thk = 0.5, 0.06, 0.05
    notch_w = max(W_mm * 0.08, 1.2)
    line_w = notch_w * 1.4

    sub_y0, sub_y1 = -L_mm/2 - sub_margin, L_mm/2 + sub_margin
    sub_x0, sub_x1 = -W_mm/2 - sub_margin, W_mm/2 + sub_margin

    ground = box_mesh((-W_mm/2 - gnd_margin, W_mm/2 + gnd_margin),
                       (-L_mm/2 - gnd_margin, L_mm/2 + gnd_margin),
                       (-gnd_thk, 0), color="#c9a227", opacity=1.0, name="Ground plane")
    substrate = box_mesh((sub_x0, sub_x1), (sub_y0, sub_y1), (0, h_mm),
                          color="#4caf6a", opacity=0.9, name="Substrate")
    patch = box_mesh((-W_mm/2, W_mm/2), (-L_mm/2, L_mm/2),
                      (h_mm, h_mm + patch_thk), color="#f4d942", opacity=1.0, name="Copper Patch")
    notch = box_mesh((-notch_w/2, notch_w/2), (-L_mm/2, -L_mm/2 + y0_mm),
                      (h_mm, h_mm + patch_thk * 0.5), color="#4caf6a", opacity=1.0,
                      name="Feed notch (inset)")
    feed_line = box_mesh((-line_w/2, line_w/2), (sub_y0, -L_mm/2 + y0_mm),
                          (h_mm, h_mm + line_thk), color="#f4d942", opacity=1.0,
                          name="Feed Line")
    port = box_mesh((-line_w/2, line_w/2), (sub_y0 - 0.25, sub_y0 + 0.25),
                     (0, h_mm), color="#cc0000", opacity=1.0, name="Lumped Port")

    ground_edges = box_edges((-W_mm/2 - gnd_margin, W_mm/2 + gnd_margin),
                              (-L_mm/2 - gnd_margin, L_mm/2 + gnd_margin), (-gnd_thk, 0))
    substrate_edges = box_edges((sub_x0, sub_x1), (sub_y0, sub_y1), (0, h_mm))
    patch_edges = box_edges((-W_mm/2, W_mm/2), (-L_mm/2, L_mm/2), (h_mm, h_mm + patch_thk))
    notch_edges = box_edges((-notch_w/2, notch_w/2), (-L_mm/2, -L_mm/2 + y0_mm),
                             (h_mm, h_mm + patch_thk * 0.5))
    feed_line_edges = box_edges((-line_w/2, line_w/2), (sub_y0, -L_mm/2 + y0_mm),
                                 (h_mm, h_mm + line_thk))
    port_edges = box_edges((-line_w/2, line_w/2), (sub_y0 - 0.25, sub_y0 + 0.25),
                            (0, h_mm), color="#660000")

    dim_offset = max(W_mm, L_mm) * 0.18
    dim_z = h_mm + patch_thk + 1.0
    w_dim = go.Scatter3d(
        x=[-W_mm/2, W_mm/2], y=[L_mm/2 + dim_offset, L_mm/2 + dim_offset], z=[dim_z, dim_z],
        mode="lines+text", line=dict(color="black", width=3),
        text=["", f"W={W_mm:.1f}mm"], textposition="top center", showlegend=False,
    )
    l_dim = go.Scatter3d(
        x=[-W_mm/2 - dim_offset, -W_mm/2 - dim_offset], y=[-L_mm/2, L_mm/2], z=[dim_z, dim_z],
        mode="lines+text", line=dict(color="black", width=3),
        text=["", f"L={L_mm:.1f}mm"], textposition="middle right", showlegend=False,
    )

    fig = go.Figure(data=[
        ground, ground_edges,
        substrate, substrate_edges,
        patch, patch_edges,
        notch, notch_edges,
        feed_line, feed_line_edges,
        port, port_edges,
        w_dim, l_dim,
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="W (mm)", backgroundcolor="rgb(245,245,245)",
                       gridcolor="#bbbbbb", zerolinecolor="#888888", showbackground=True),
            yaxis=dict(title="L (mm)", backgroundcolor="rgb(245,245,245)",
                       gridcolor="#bbbbbb", zerolinecolor="#888888", showbackground=True),
            zaxis=dict(title="height (mm)", backgroundcolor="rgb(250,250,250)",
                       gridcolor="#cccccc", zerolinecolor="#888888", showbackground=True),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.9, z=1.3)),
        ),
        margin=dict(l=0, r=0, t=10, b=0), height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.0, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Radiation pattern: cavity-model (two-slot) closed-form approximation,
# driven by this design's actual L, W, and resonant frequency.
# ---------------------------------------------------------------------------
def cavity_model_pattern(theta_rad, L_m, W_m, freq_hz):
    c = 3e8
    k0 = 2 * np.pi * freq_hz / c
    sin_t = np.sin(theta_rad)

    e_plane = np.cos((k0 * L_m / 2) * sin_t)

    x = (k0 * W_m / 2) * sin_t
    h_plane = np.where(np.abs(x) < 1e-6, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
    h_plane = h_plane * np.cos(theta_rad)

    return np.abs(e_plane), np.abs(h_plane)


def build_pattern_cuts_2d(L_m, W_m, freq_hz):
    theta_deg = np.linspace(-90, 90, 181)
    theta_rad = np.radians(theta_deg)
    e_plane, h_plane = cavity_model_pattern(theta_rad, L_m, W_m, freq_hz)
    e_db = np.clip(20 * np.log10(np.clip(e_plane, 1e-3, None)), -30, 0) + 30
    h_db = np.clip(20 * np.log10(np.clip(h_plane, 1e-3, None)), -30, 0) + 30

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=e_db, theta=theta_deg, mode="lines",
                                   name="E-plane", line=dict(color="#2d5f8a", width=3)))
    fig.add_trace(go.Scatterpolar(r=h_db, theta=theta_deg, mode="lines",
                                   name="H-plane", line=dict(color="#cf222e", width=3)))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 30], showticklabels=False, ticks=""),
                   angularaxis=dict(direction="clockwise")),
        template="plotly_white", height=380, margin=dict(t=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def build_pattern_3d(L_m, W_m, freq_hz):
    theta = np.linspace(0, np.pi / 2, 50)
    phi = np.linspace(0, 2 * np.pi, 72)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    e_plane, h_plane = cavity_model_pattern(theta_grid, L_m, W_m, freq_hz)
    power = (h_plane ** 2) * np.cos(phi_grid) ** 2 + (e_plane ** 2) * np.sin(phi_grid) ** 2
    r = np.sqrt(np.clip(power, 0, None))

    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=r,
                                      colorscale="Viridis", showscale=False)])
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="", yaxis_title="", zaxis_title=""),
        margin=dict(l=0, r=0, t=10, b=0), height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Real HFSS far-field pattern plots (theta/phi/gain grid from
# simulate_hfss_with_pattern) -- NOT the cavity-model approximation above.
# ---------------------------------------------------------------------------
def build_real_pattern_cuts_2d(pattern):
    theta_deg = pattern["theta_deg"]
    phi_deg = pattern["phi_deg"]
    gain_db = pattern["gain_db"]  # shape (n_theta, n_phi)

    phi0_idx = int(np.argmin(np.abs(phi_deg - 0)))
    phi90_idx = int(np.argmin(np.abs(phi_deg - 90)))

    e_cut = gain_db[:, phi0_idx]
    h_cut = gain_db[:, phi90_idx]

    floor = -30
    e_plot = np.clip(e_cut, floor, None) - floor
    h_plot = np.clip(h_cut, floor, None) - floor

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=e_plot, theta=theta_deg, mode="lines",
                                   name="phi=0° (E-plane)", line=dict(color="#2d5f8a", width=3)))
    fig.add_trace(go.Scatterpolar(r=h_plot, theta=theta_deg, mode="lines",
                                   name="phi=90° (H-plane)", line=dict(color="#cf222e", width=3)))
    fig.update_layout(
        polar=dict(radialaxis=dict(showticklabels=False, ticks=""),
                   angularaxis=dict(direction="clockwise")),
        template="plotly_white", height=380, margin=dict(t=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def build_real_pattern_3d(pattern):
    theta_deg = pattern["theta_deg"]
    phi_deg = pattern["phi_deg"]
    gain_db = pattern["gain_db"]

    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    theta_grid, phi_grid = np.meshgrid(theta_rad, phi_rad, indexing="ij")

    floor = -20
    gain_lin = np.clip(gain_db - floor, 0, None)

    x = gain_lin * np.sin(theta_grid) * np.cos(phi_grid)
    y = gain_lin * np.sin(theta_grid) * np.sin(phi_grid)
    z = gain_lin * np.cos(theta_grid)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=gain_db,
                                      colorscale="Viridis", showscale=True,
                                      colorbar=dict(title="dBi"))])
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="", yaxis_title="", zaxis_title=""),
        margin=dict(l=0, r=0, t=10, b=0), height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Target Spec")
    freq_ghz = st.slider("Frequency (GHz)", 0.5, 10.0, 2.4, 0.1)
    min_s11_db = st.slider("Min S11 (dB)", -40, -5, -15, 1)
    use_gain = st.checkbox("Also target peak gain")
    gain_dbi = st.slider("Target gain (dBi)", 0.0, 10.0, 5.0, 0.5) if use_gain else None
    st.divider()
    run_btn = st.button("🚀 Auto-Design (inverse search)", use_container_width=True, type="primary")
    st.caption("Auto-relaxes the S11 and/or gain targets if your combined spec isn't achievable, "
               "and reports the closest achievable design.")
    st.caption(f"HFSS-calibrated for {CALIB_MIN:.1f}–{CALIB_MAX:.1f} GHz.")
    st.divider()
    if HFSS_AVAILABLE:
        st.caption("✅ Local AEDT/pyaedt detected — HFSS validation tab available.")
    else:
        st.caption("⚪ pyaedt unavailable in this environment — HFSS tab disabled.")

for key, default in [("ws_L", 29.4), ("ws_W", 38.0), ("ws_h", 1.6),
                      ("ws_eps", 4.4), ("ws_y0", 7.9)]:
    if key not in st.session_state:
        st.session_state[key] = default

if run_btn:
    with st.spinner("Running inverse search..."):
        result, gt, effective_target, effective_gain, attempts = auto_relax(freq_ghz, min_s11_db, gain_dbi)
    design = result["snapped_design"]
    st.session_state["ws_L"] = round(design.L * 1e3, 3)
    st.session_state["ws_W"] = round(design.W * 1e3, 3)
    st.session_state["ws_h"] = round(design.h * 1e3, 3)
    st.session_state["ws_eps"] = round(design.eps_r, 3)
    st.session_state["ws_y0"] = round(design.y0 * 1e3, 3)
    st.session_state["auto_result"] = result
    st.session_state["auto_gt"] = gt
    st.session_state["auto_effective_target"] = effective_target
    st.session_state["auto_effective_gain"] = effective_gain
    st.session_state["auto_attempts"] = attempts
    st.session_state["auto_relaxed"] = (effective_target != min_s11_db) or (effective_gain != gain_dbi)
    st.session_state["auto_orig_target"] = min_s11_db
    st.session_state["auto_orig_gain"] = gain_dbi
    st.session_state["auto_freq_ghz"] = freq_ghz
    st.session_state.pop("hfss_result", None)
    st.session_state.pop("hfss_pattern", None)

tab1, tab2, tab3 = st.tabs(["🧊 Design Workspace", "🔍 Raw Candidates", "🔬 HFSS Validation"])

# ---------------------------------------------------------------------------
# TAB 1 — CAD-style split workspace (this IS the manual fine-tune surface:
# the left accordion sliders drive the live 3D model + S11/VSWR/Radiation
# panels on the right in real time)
# ---------------------------------------------------------------------------
with tab1:
    if "auto_relaxed" in st.session_state and st.session_state["auto_relaxed"]:
        gt0 = st.session_state["auto_gt"]
        tgt0 = st.session_state["auto_effective_target"]
        gain0 = st.session_state["auto_effective_gain"]
        orig0 = st.session_state["auto_orig_target"]
        orig_gain0 = st.session_state["auto_orig_gain"]
        f0 = st.session_state["auto_freq_ghz"]

        s11_ok = gt0.min_s11_db <= tgt0
        gain_ok = (gain0 is None) or (gt0.peak_gain_dbi >= gain0)

        msgs = []
        if tgt0 != orig0:
            msgs.append(f"S11 target {orig0}dB → relaxed to {tgt0}dB (achieved {gt0.min_s11_db:.1f}dB)")
        if orig_gain0 is not None and gain0 != orig_gain0:
            msgs.append(f"Gain target {orig_gain0}dBi → relaxed to {gain0:.1f}dBi (achieved {gt0.peak_gain_dbi:.2f}dBi)")
        msg_text = "; ".join(msgs) if msgs else "target adjusted"

        if s11_ok and gain_ok:
            st.warning(f"⚠️ Original target not fully achievable at {f0}GHz — auto-relaxed. {msg_text}. "
                       "Workspace below is pre-loaded with this result; edit freely.")
        else:
            st.error(f"❌ Could not meet target even after relaxing as far as allowed. "
                     f"Achieved S11={gt0.min_s11_db:.1f}dB, Gain={gt0.peak_gain_dbi:.2f}dBi. "
                     "Workspace pre-loaded with closest attempt.")

        with st.expander("Auto-relax attempt log"):
            for a in st.session_state["auto_attempts"]:
                tick = "✅" if a["met"] else "—"
                gain_str = (f", gain target {a['target_gain_dbi']}dBi → achieved {a['achieved_gain_dbi']:.2f}dBi"
                            if a["target_gain_dbi"] is not None else "")
                st.write(f"{tick} S11 target {a['target_s11_db']}dB → achieved {a['achieved_s11_db']:.2f}dB{gain_str}")

    left, center, right = st.columns([1, 1.7, 1.3])

    with left:
        st.markdown("#### 📋 Parameters (manual fine-tune)")
        with st.expander("Substrate", expanded=True):
            st.slider("εr (permittivity)", 1.5, 12.0, key="ws_eps", step=0.05)
            st.slider("Height h (mm)", 0.1, 6.0, key="ws_h", step=0.01)
        with st.expander("Patch Geometry", expanded=True):
            st.slider("Length L (mm)", 2.0, 100.0, key="ws_L", step=0.1)
            st.slider("Width W (mm)", 2.0, 100.0, key="ws_W", step=0.1)
        with st.expander("Feed", expanded=True):
            y0_max = max(st.session_state["ws_L"] / 2 - 0.1, 0.2)
            st.session_state["ws_y0"] = min(st.session_state["ws_y0"], y0_max)
            st.slider("Inset depth y0 (mm)", 0.1, float(y0_max), key="ws_y0", step=0.05)
            feed_type = st.selectbox("Feed type", [
                "Coaxial probe (inset-fed) — solver supports this",
                "Microstrip line — NOT modeled by current solver",
            ])
            if "NOT modeled" in feed_type:
                st.caption("⚠️ Informational only — all physics shown uses the coaxial inset model.")

        live_design = DesignParams(
            L=st.session_state["ws_L"] / 1e3, W=st.session_state["ws_W"] / 1e3,
            h=st.session_state["ws_h"] / 1e3, eps_r=st.session_state["ws_eps"],
            y0=st.session_state["ws_y0"] / 1e3,
        )
        freq_grid = np.linspace(0.5e9, 10e9, 600)
        try:
            live_gt = simulate(live_design, freq_grid)
            sim_ok = True
        except Exception as e:
            sim_ok = False
            st.error(f"Invalid geometry: {e}")

        if sim_ok:
            err_pct, extrapolated = estimate_accuracy(live_gt.resonant_freq_hz / 1e9)
            st.markdown(accuracy_badge_html(err_pct, extrapolated), unsafe_allow_html=True)

            s11_ok_live = live_gt.min_s11_db <= min_s11_db
            status_class = "status-pass" if s11_ok_live else "status-fail"
            status_text = "✅ MEETS sidebar S11 target" if s11_ok_live else "⚠️ does not meet sidebar S11 target"
            st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)

            if gain_dbi is not None:
                gain_ok_live = live_gt.peak_gain_dbi >= gain_dbi
                st.caption("✅ meets sidebar gain target" if gain_ok_live
                           else "⚠️ below sidebar gain target — try Auto-Design or hand-tune geometry")

    with center:
        st.markdown("#### 🧊 3D Geometry (live)")
        st.plotly_chart(
            build_3d_view(st.session_state["ws_L"], st.session_state["ws_W"],
                          st.session_state["ws_h"], st.session_state["ws_y0"]),
            use_container_width=True,
        )
        st.caption("Inset-fed microstrip patch antenna: copper patch (yellow) on substrate (green) over "
                   "a ground plane, with the feed line trace running from the substrate edge into the "
                   "inset notch at depth y0, and the lumped port where it terminates — matching the "
                   "standard inset-fed topology used by the analytical model and HFSS runs in this project.")

    with right:
        st.markdown("#### 📈 Physics (analytical model)")
        if sim_ok:
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("f₀", f"{live_gt.resonant_freq_hz/1e9:.3f} GHz")
            pc2.metric("S11", f"{live_gt.min_s11_db:.1f} dB")
            pc3.metric("Gain", f"{live_gt.peak_gain_dbi:.2f} dBi")

            pt1, pt2, pt3 = st.tabs(["S11", "VSWR", "Radiation Pattern"])
            with pt1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=live_gt.freq_hz/1e9, y=live_gt.s11_db,
                                          mode="lines", line=dict(color="#2d5f8a", width=3),
                                          fill="tozeroy", fillcolor="rgba(45,95,138,0.08)"))
                fig.add_hline(y=min_s11_db, line_dash="dash", line_color="#cf222e")
                fig.add_vline(x=freq_ghz, line_dash="dot", line_color="gray")
                fig.update_layout(template="plotly_white", height=320,
                                   xaxis_title="GHz", yaxis_title="dB", margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
            with pt2:
                vswr = compute_vswr(live_gt.s11_db)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=live_gt.freq_hz/1e9, y=vswr,
                                          mode="lines", line=dict(color="#8a4b2d", width=3)))
                fig.add_hline(y=2.0, line_dash="dash", line_color="#cf222e",
                              annotation_text="VSWR=2 (common spec)")
                fig.update_layout(template="plotly_white", height=320,
                                   xaxis_title="GHz", yaxis_title="VSWR", margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)
            with pt3:
                st.plotly_chart(build_pattern_cuts_2d(live_design.L, live_design.W,
                                                       live_gt.resonant_freq_hz), use_container_width=True)
                st.plotly_chart(build_pattern_3d(live_design.L, live_design.W,
                                                  live_gt.resonant_freq_hz), use_container_width=True)
                st.caption("Cavity-model (two-slot) closed-form E-plane/H-plane patterns, computed from "
                           "this design's actual L, W and resonant frequency (Balanis two-slot "
                           "approximation). The 3D surface interpolates between these two cuts. This is "
                           "a standard analytical approximation, NOT a full-wave HFSS far-field result.")

    st.caption("⚠️ Workspace physics use the analytical model (fast, approximate). "
               "For ground truth, run **HFSS Validation**.")

# ---------------------------------------------------------------------------
# TAB 2 — raw candidates from the last auto-search
# ---------------------------------------------------------------------------
with tab2:
    if "auto_result" not in st.session_state:
        st.info("Run an auto-design search in the sidebar first.")
    else:
        st.caption("Top surrogate-predicted candidates before substrate-snapping:")
        for i, cand in enumerate(st.session_state["auto_result"]["candidates"]):
            with st.expander(f"Candidate {i+1} — loss={cand['loss']:.5f}"):
                st.json(cand)

# ---------------------------------------------------------------------------
# TAB 3 — real HFSS validation of the current workspace design, with the
# option to watch it build and solve live in the AEDT GUI, plus a real
# far-field radiation pattern pulled from the solved model.
# ---------------------------------------------------------------------------
with tab3:
    st.caption("Runs the **actual** HFSS full-wave simulation on the current workspace geometry "
               "(left panel values in Design Workspace) — ground truth, not the analytical approximation.")

    if not HFSS_AVAILABLE:
        st.error("pyaedt / hfss_model unavailable in this environment. Requires local Windows + "
                 "AEDT 2022 R1 + pyaedt==0.6.70. Not available on Colab/cloud.")
    else:
        hfss_design = DesignParams(
            L=st.session_state["ws_L"] / 1e3, W=st.session_state["ws_W"] / 1e3,
            h=st.session_state["ws_h"] / 1e3, eps_r=st.session_state["ws_eps"],
            y0=st.session_state["ws_y0"] / 1e3,
        )
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write(f"**Design to validate:** L={hfss_design.L*1e3:.2f}mm, W={hfss_design.W*1e3:.2f}mm, "
                 f"h={hfss_design.h*1e3:.3f}mm, εr={hfss_design.eps_r:.2f}, y0={hfss_design.y0*1e3:.2f}mm")
        st.markdown('</div>', unsafe_allow_html=True)

        watch_live = st.checkbox(
            "👁️ Watch it build live in HFSS (opens the AEDT window — airbox, substrate, "
            "ground, patch, feed, port, and far-field setup built and solved visibly)",
            value=True,
        )
        if watch_live:
            st.warning("⏱️ Takes approx. 10-15 minutes (full angular far-field sweep included), uses a "
                       "local AEDT license, and **opens a visible AEDT window** on this machine. "
                       "Confirm no zombie `ansysedt.exe` process is already running first.")
        else:
            st.warning("⏱️ Takes approx. 10-15 minutes, uses a local AEDT license, runs headless "
                       "(no visible window). Confirm no zombie `ansysedt.exe` process is already running first.")
        confirm = st.checkbox("I understand this will take several minutes and uses local AEDT.")
        hfss_btn = st.button("🔬 Run HFSS Validation", disabled=not confirm, type="primary")

        if hfss_btn:
            hfss_freq_hz = np.linspace(freq_ghz * 0.8e9, freq_ghz * 1.2e9, 81)
            spinner_text = ("Building and solving in the visible AEDT window — watch it appear "
                             "on your screen (10-15 min)..." if watch_live else
                             "Running HFSS full-wave simulation headless (10-15 min)...")
            try:
                with st.spinner(spinner_text):
                    hfss_gt_new, pattern_new = simulate_hfss_with_pattern(
                        hfss_design, hfss_freq_hz, non_graphical=not watch_live,
                        keep_open=watch_live,
                    )
                st.session_state["hfss_result"] = hfss_gt_new
                st.session_state["hfss_pattern"] = pattern_new
                st.session_state["hfss_design_used"] = hfss_design
                st.session_state["hfss_target_freq_ghz"] = freq_ghz
                st.success("HFSS simulation complete.")
            except Exception as e:
                st.error(f"HFSS simulation failed: {e}")

        if "hfss_result" in st.session_state:
            hfss_gt = st.session_state["hfss_result"]
            hfss_pattern = st.session_state.get("hfss_pattern")
            hfss_design_used = st.session_state["hfss_design_used"]
            hfss_target_freq_ghz = st.session_state["hfss_target_freq_ghz"]
            freq_err_pct = abs(hfss_gt.resonant_freq_hz - hfss_target_freq_ghz * 1e9) / (hfss_target_freq_ghz * 1e9) * 100

            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("HFSS f₀", f"{hfss_gt.resonant_freq_hz/1e9:.3f} GHz", f"{freq_err_pct:.1f}% error")
            hc2.metric("HFSS S11", f"{hfss_gt.min_s11_db:.1f} dB")
            hfss_vswr_min = compute_vswr(hfss_gt.min_s11_db)
            hc3.metric("HFSS VSWR (at min S11)", f"{float(hfss_vswr_min):.2f}")
            hc4.metric("HFSS Gain", f"{hfss_gt.peak_gain_dbi:.2f} dBi")

            with st.expander("🔧 Debug: raw HFSS result values"):
                st.write({
                    "resonant_freq_hz": hfss_gt.resonant_freq_hz,
                    "min_s11_db": hfss_gt.min_s11_db,
                    "peak_gain_dbi": hfss_gt.peak_gain_dbi,
                    "freq_hz array length": len(hfss_gt.freq_hz),
                    "s11_db array length": len(hfss_gt.s11_db),
                    "pattern available": hfss_pattern is not None,
                })

            vswr_fig = go.Figure()
            vswr_fig.add_trace(go.Scatter(x=hfss_gt.freq_hz/1e9, y=compute_vswr(hfss_gt.s11_db),
                                           mode="lines", name="VSWR (HFSS)", line=dict(color="#8a4b2d", width=3)))
            vswr_fig.add_hline(y=2.0, line_dash="dash", line_color="#cf222e", annotation_text="VSWR=2")
            vswr_fig.update_layout(title="HFSS VSWR vs Frequency", xaxis_title="GHz", yaxis_title="VSWR",
                                    template="plotly_white", height=320)
            st.plotly_chart(vswr_fig, use_container_width=True)

            analytical_gt = simulate(hfss_design_used, np.linspace(0.5e9, 10e9, 600))
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=analytical_gt.freq_hz/1e9, y=analytical_gt.s11_db,
                                       mode="lines", name="Analytical",
                                       line=dict(color="#2d5f8a", width=2, dash="dot")))
            fig3.add_trace(go.Scatter(x=hfss_gt.freq_hz/1e9, y=hfss_gt.s11_db,
                                       mode="lines", name="HFSS (ground truth)",
                                       line=dict(color="#cf222e", width=3),
                                       fill="tozeroy", fillcolor="rgba(207,34,46,0.08)"))
            fig3.update_layout(title="Analytical Prediction vs Real HFSS Result",
                                xaxis_title="GHz", yaxis_title="dB",
                                template="plotly_white", height=420)
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("#### 📡 Real HFSS Far-Field Radiation Pattern")
            if hfss_pattern is not None:
                st.caption("Actual full-wave far-field gain pattern pulled from HFSS (theta 0-180°, "
                           "phi 0-360°) at the solved resonant frequency — not the cavity-model "
                           "approximation used in the Design Workspace tab.")
                rp1, rp2 = st.columns(2)
                with rp1:
                    st.plotly_chart(build_real_pattern_cuts_2d(hfss_pattern), use_container_width=True)
                with rp2:
                    st.plotly_chart(build_real_pattern_3d(hfss_pattern), use_container_width=True)
            else:
                st.warning("Far-field pattern extraction failed for this run (see console/log output). "
                           f"Scalar peak gain is still valid: {hfss_gt.peak_gain_dbi:.2f} dBi at boresight.")