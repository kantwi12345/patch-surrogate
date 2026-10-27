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
    from src.sim.hfss_model import simulate_hfss
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
        font-size: 1.4rem !important;
        white-space: normal !important;
        overflow: visible !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #444 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📡 Patch Antenna Inverse Design</h1>
    <p>Specify a target frequency and spec — get a manufacturable geometry back in milliseconds.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HFSS calibration data (from M5_RESULTS.md, 8 real full-wave validation runs).
# Used to give an honest "expected real-world frequency error" estimate,
# since the on-screen curve is the analytical model, not HFSS itself.
# ---------------------------------------------------------------------------
CALIBRATION_FREQ_GHZ = np.array([1.2, 1.8, 2.4, 3.0, 3.5, 4.2, 5.0, 5.8])
CALIBRATION_FREQ_ERR_PCT = np.array([4.1, 1.5, 0.9, 0.7, 2.7, 6.7, 11.9, 3.4])
CALIB_MIN, CALIB_MAX = CALIBRATION_FREQ_GHZ.min(), CALIBRATION_FREQ_GHZ.max()


def estimate_accuracy(freq_ghz):
    """
    Interpolate expected real-world (HFSS) frequency error % at a given
    target frequency, from the 8 known calibration points. Returns
    (error_pct, extrapolated_bool). Outside the calibrated 1.2-5.8GHz
    range, we extrapolate flatly from the nearest edge point and flag it.
    """
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
    extra = " (extrapolated — outside calibrated 1.2-5.8GHz range, use with caution)" if extrapolated else ""
    return (
        f'<span class="accuracy-badge" style="background:{color}20;color:{color};">'
        f'{label} confidence — expected freq error ≈ ±{err_pct:.1f}%{extra}</span>'
    )


def auto_relax(freq_ghz, min_s11_db, gain_dbi, max_steps=6, step_db=2):
    """
    Try the requested S11 target first. If not met, progressively relax
    (move toward -5dB, the least-strict allowed) and retry, up to max_steps
    times. Each retry is a full design_from_spec() call (milliseconds,
    surrogate-only, no HFSS). Returns the best result found plus a log of
    every attempt so the UI can explain what happened.
    """
    attempts = []
    target = min_s11_db
    result, gt = None, None
    for i in range(max_steps):
        result = design_from_spec(
            target_freq_hz=freq_ghz * 1e9,
            target_min_s11_db=target,
            target_gain_dbi=gain_dbi,
        )
        gt = result["ground_truth"]
        met = gt.min_s11_db <= target
        attempts.append({"target_s11_db": target, "achieved_s11_db": gt.min_s11_db, "met": met})
        if met:
            return result, gt, target, attempts
        if target >= -5:
            break
        target = min(target + step_db, -5)
    return result, gt, target, attempts


with st.sidebar:
    st.header("⚙️ Target Spec")
    freq_ghz = st.slider("Frequency (GHz)", 0.5, 10.0, 2.4, 0.1)
    min_s11_db = st.slider("Min S11 (dB)", -40, -5, -15, 1)
    use_gain = st.checkbox("Also target peak gain")
    gain_dbi = st.slider("Target gain (dBi)", 0.0, 10.0, 5.0, 0.5) if use_gain else None
    st.divider()
    run_btn = st.button("🚀 Design Antenna", use_container_width=True, type="primary")
    st.caption("If the target isn't achievable, the app auto-relaxes the S11 threshold "
               "and reports the closest achievable spec.")
    st.caption(f"HFSS-calibrated for {CALIB_MIN:.1f}–{CALIB_MAX:.1f} GHz. Outside that range, "
               "accuracy estimates are extrapolated.")
    st.divider()
    if HFSS_AVAILABLE:
        st.caption("✅ Local AEDT/pyaedt detected — real HFSS validation available in the fourth tab.")
    else:
        st.caption("⚪ pyaedt not available in this environment — HFSS validation tab disabled "
                   "(expected on Colab/cloud; works locally with AEDT 2022 R1 installed).")

if run_btn:
    with st.spinner("Optimizing geometry..."):
        result, gt, effective_target, attempts = auto_relax(freq_ghz, min_s11_db, gain_dbi)
    spec_met = gt.min_s11_db <= effective_target
    relaxed = effective_target != min_s11_db
    st.session_state["result"] = result
    st.session_state["gt"] = gt
    st.session_state["effective_target"] = effective_target
    st.session_state["attempts"] = attempts
    st.session_state["spec_met"] = spec_met
    st.session_state["relaxed"] = relaxed
    st.session_state["orig_target"] = min_s11_db
    st.session_state["freq_ghz"] = freq_ghz
    st.session_state["gain_dbi"] = gain_dbi
    st.session_state.pop("hfss_result", None)  # invalidate any stale HFSS run

has_result = "result" in st.session_state

tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Design", "🔍 Raw Candidates", "🛠️ Manual Fine-Tune", "🔬 HFSS Validation"
])

with tab1:
    if not has_result:
        st.info("👈 Set your target spec in the sidebar and click **Design Antenna** to get started.")
    else:
        result = st.session_state["result"]
        gt = st.session_state["gt"]
        effective_target = st.session_state["effective_target"]
        spec_met = st.session_state["spec_met"]
        relaxed = st.session_state["relaxed"]
        orig_target = st.session_state["orig_target"]
        freq_ghz_r = st.session_state["freq_ghz"]
        gain_dbi_r = st.session_state["gain_dbi"]
        design = result["snapped_design"]

        err_pct, extrapolated = estimate_accuracy(freq_ghz_r)
        st.markdown(accuracy_badge_html(err_pct, extrapolated), unsafe_allow_html=True)
        st.caption("Confidence estimate interpolated from 8 real HFSS full-wave validation runs "
                   "(see M5_RESULTS.md) — NOT measured on this specific design. "
                   "Run the **HFSS Validation** tab to measure it directly.")

        if relaxed:
            if spec_met:
                st.warning(
                    f"⚠️ Target **{orig_target}dB** was not achievable at {freq_ghz_r}GHz with the "
                    f"available substrates. Auto-relaxed to **{effective_target}dB** — closest "
                    f"achievable design shown below (achieved {gt.min_s11_db:.1f}dB)."
                )
            else:
                st.error(
                    f"❌ Even after relaxing from {orig_target}dB up to the loosest allowed "
                    f"({effective_target}dB), the target could not be met. Best achieved: "
                    f"{gt.min_s11_db:.1f}dB. Try a different frequency or substrate range."
                )
            with st.expander("Auto-relax attempt log"):
                for a in st.session_state["attempts"]:
                    tick = "✅" if a["met"] else "—"
                    st.write(f"{tick} target {a['target_s11_db']}dB → achieved {a['achieved_s11_db']:.2f}dB")

        col_left, col_right = st.columns([1, 1.3])

        with col_left:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader(f"Substrate: {result['substrate_name']}")
            c1, c2 = st.columns(2)
            c1.metric("Length (L)", f"{design.L*1e3:.2f} mm")
            c1.metric("Width (W)", f"{design.W*1e3:.2f} mm")
            c1.metric("Height (h)", f"{design.h*1e3:.3f} mm")
            c2.metric("εr", f"{design.eps_r:.2f}")
            c2.metric("Feed inset (y0)", f"{design.y0*1e3:.2f} mm")
            st.markdown('</div>', unsafe_allow_html=True)

            status_class = "status-pass" if spec_met else "status-fail"
            status_text = "✅ SPEC MET" if spec_met else "⚠️ SPEC NOT MET"
            st.markdown(f'<div class="result-card"><span class="{status_class}">{status_text}</span></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Resonant Freq", f"{gt.resonant_freq_hz/1e9:.3f} GHz", f"{(gt.resonant_freq_hz/1e9 - freq_ghz_r):+.3f} GHz")
            c2.metric("Min S11", f"{gt.min_s11_db:.1f} dB")
            c3.metric("Peak Gain", f"{gt.peak_gain_dbi:.2f} dBi")

            if gain_dbi_r is not None:
                gain_ok = gt.peak_gain_dbi >= gain_dbi_r
                note = "✅ gain target met" if gain_ok else f"⚠️ gain target ({gain_dbi_r} dBi) not met — not auto-correctable, try Manual Fine-Tune"
                st.caption(note)

        with col_right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gt.freq_hz / 1e9, y=gt.s11_db,
                mode="lines", name="S11 (analytical)", line=dict(color="#2d5f8a", width=3),
                fill="tozeroy", fillcolor="rgba(45,95,138,0.08)",
            ))
            fig.add_hline(y=effective_target, line_dash="dash", line_color="#cf222e",
                          annotation_text=f"Target {effective_target}dB", annotation_position="bottom right")
            fig.add_vline(x=freq_ghz_r, line_dash="dot", line_color="gray",
                          annotation_text="Target freq", annotation_position="top")
            band_lo = freq_ghz_r * (1 - err_pct / 100)
            band_hi = freq_ghz_r * (1 + err_pct / 100)
            fig.add_vrect(x0=band_lo, x1=band_hi, fillcolor="orange", opacity=0.12, line_width=0,
                          annotation_text="expected real-world freq range", annotation_position="top left")
            fig.update_layout(
                title="S11 vs Frequency (analytical model)",
                xaxis_title="Frequency (GHz)", yaxis_title="S11 (dB)",
                template="plotly_white", height=420, margin=dict(t=50),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption("⚠️ Analytical self-consistency check only — y0 is HFSS-calibrated but intentionally "
                   "detuned from this model's own optimum to match physical reality. "
                   "Run the **HFSS Validation** tab for true full-wave validation.")

with tab2:
    if not has_result:
        st.info("Run a design first to see candidates.")
    else:
        st.caption("Top surrogate-predicted candidates before substrate-snapping (from the final accepted attempt):")
        for i, cand in enumerate(st.session_state["result"]["candidates"]):
            with st.expander(f"Candidate {i+1} — loss={cand['loss']:.5f}"):
                st.json(cand)

with tab3:
    st.caption("Directly set geometry and see the analytical response — use this when auto-search "
               "can't hit your target and you want to hand-tune (e.g. trade gain for bandwidth).")

    if has_result:
        seed = st.session_state["result"]["snapped_design"]
        default_L, default_W, default_h, default_eps, default_y0 = (
            seed.L * 1e3, seed.W * 1e3, seed.h * 1e3, seed.eps_r, seed.y0 * 1e3,
        )
    else:
        default_L, default_W, default_h, default_eps, default_y0 = 29.4, 38.0, 1.6, 4.4, 7.9

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        m_L = st.slider("Length L (mm)", 2.0, 100.0, float(default_L), 0.1)
        m_W = st.slider("Width W (mm)", 2.0, 100.0, float(default_W), 0.1)
    with mc2:
        m_h = st.slider("Height h (mm)", 0.1, 6.0, float(default_h), 0.01)
        m_eps = st.slider("εr", 1.5, 12.0, float(default_eps), 0.05)
    with mc3:
        m_y0_max = max(m_L / 2 - 0.1, 0.2)
        m_y0 = st.slider("Feed inset y0 (mm)", 0.1, float(m_y0_max), min(float(default_y0), float(m_y0_max)), 0.05)

    manual_design = DesignParams(
        L=m_L / 1e3, W=m_W / 1e3, h=m_h / 1e3, eps_r=m_eps, y0=m_y0 / 1e3,
    )
    freq_grid = np.linspace(0.5e9, 10e9, 600)

    try:
        manual_gt = simulate(manual_design, freq_grid)
        m_err_pct, m_extrapolated = estimate_accuracy(manual_gt.resonant_freq_hz / 1e9)
        st.markdown(accuracy_badge_html(m_err_pct, m_extrapolated), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Resonant Freq", f"{manual_gt.resonant_freq_hz/1e9:.3f} GHz")
        c2.metric("Min S11", f"{manual_gt.min_s11_db:.1f} dB")
        c3.metric("Peak Gain", f"{manual_gt.peak_gain_dbi:.2f} dBi")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=manual_gt.freq_hz / 1e9, y=manual_gt.s11_db,
            mode="lines", name="S11", line=dict(color="#8a4b2d", width=3),
            fill="tozeroy", fillcolor="rgba(138,75,45,0.08)",
        ))
        fig2.update_layout(
            title="Manual Geometry — S11 vs Frequency (analytical model)",
            xaxis_title="Frequency (GHz)", yaxis_title="S11 (dB)",
            template="plotly_white", height=420, margin=dict(t=50),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Could not simulate this geometry: {e}")

with tab4:
    st.caption("Run the **actual** HFSS full-wave simulation on the current design — ground truth, "
               "not the analytical approximation shown in the other tabs.")

    if not HFSS_AVAILABLE:
        st.error("pyaedt / hfss_model could not be imported in this environment. This feature "
                 "requires a local Windows machine with AEDT 2022 R1 installed, licensed, and "
                 "pyaedt==0.6.70. It is not available on Colab or a cloud deployment of this app.")
    elif not has_result:
        st.info("Run a design in the sidebar first, then come back here to validate it with HFSS.")
    else:
        design = st.session_state["result"]["snapped_design"]
        freq_ghz_r = st.session_state["freq_ghz"]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write(f"**Design to validate:** L={design.L*1e3:.2f}mm, W={design.W*1e3:.2f}mm, "
                 f"h={design.h*1e3:.3f}mm, εr={design.eps_r:.2f}, y0={design.y0*1e3:.2f}mm")
        st.markdown('</div>', unsafe_allow_html=True)

        st.warning(
            "⏱️ This runs a real full-wave simulation and typically takes **5-10 minutes**. "
            "It requires a free AEDT license slot on this machine. Before running, make sure "
            "no leftover `ansysedt.exe` process is already running (check Task Manager / "
            "`Get-Process ansysedt`), or this may fail or hang."
        )
        confirm = st.checkbox("I understand this will take several minutes and uses local AEDT.")
        hfss_btn = st.button("🔬 Run HFSS Validation", disabled=not confirm, type="primary")

        if hfss_btn:
            freq_hz = np.linspace(freq_ghz_r * 0.8e9, freq_ghz_r * 1.2e9, 81)
            try:
                with st.spinner("Running HFSS full-wave simulation... do not close this tab (5-10 min)."):
                    hfss_gt = simulate_hfss(design, freq_hz, non_graphical=True)
                st.session_state["hfss_result"] = hfss_gt
                st.success("HFSS simulation complete.")
            except Exception as e:
                st.error(f"HFSS simulation failed: {e}")
                st.caption("Common causes: AEDT not installed/licensed, a zombie ansysedt.exe process "
                           "already running, or pyaedt version mismatch (requires 0.6.70).")

        if "hfss_result" in st.session_state:
            hfss_gt = st.session_state["hfss_result"]
            analytical_gt = st.session_state["gt"]

            freq_err_pct = abs(hfss_gt.resonant_freq_hz - freq_ghz_r * 1e9) / (freq_ghz_r * 1e9) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("HFSS Resonant Freq", f"{hfss_gt.resonant_freq_hz/1e9:.3f} GHz",
                      f"{freq_err_pct:.1f}% error vs target")
            c2.metric("HFSS Min S11", f"{hfss_gt.min_s11_db:.1f} dB")
            c3.metric("HFSS Peak Gain", f"{hfss_gt.peak_gain_dbi:.2f} dBi")

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=analytical_gt.freq_hz / 1e9, y=analytical_gt.s11_db,
                mode="lines", name="Analytical (predicted)",
                line=dict(color="#2d5f8a", width=2, dash="dot"),
            ))
            fig3.add_trace(go.Scatter(
                x=hfss_gt.freq_hz / 1e9, y=hfss_gt.s11_db,
                mode="lines", name="HFSS (ground truth)",
                line=dict(color="#cf222e", width=3),
                fill="tozeroy", fillcolor="rgba(207,34,46,0.08)",
            ))
            fig3.add_vline(x=freq_ghz_r, line_dash="dot", line_color="gray",
                          annotation_text="Target freq", annotation_position="top")
            fig3.update_layout(
                title="Analytical Prediction vs Real HFSS Result",
                xaxis_title="Frequency (GHz)", yaxis_title="S11 (dB)",
                template="plotly_white", height=450, margin=dict(t=50),
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.caption("This is the real measurement of pipeline accuracy for this specific design — "
                       "compare against the calibration-based estimate shown in the Design tab.")