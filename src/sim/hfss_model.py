"""
M4 — Real HFSS simulation via pyaedt, built incrementally.

STAGE 1: geometry only - DONE, verified.
STAGE 2: lumped port - DONE, verified.
STAGE 3: airbox + radiation boundary - DONE, verified.
STAGE 4: setup + frequency sweep + solve.
"""

import numpy as np
from pyaedt import Hfss

from src.sim.schema import DesignParams, SimResult
from src.sim.analytical_model import _resonant_frequency

C0 = 299_792_458.0


def _feed_line_width(h, eps_r, z0=50.0):
    A = (z0 / 60.0) * np.sqrt((eps_r + 1) / 2) + (eps_r - 1) / (eps_r + 1) * (0.23 + 0.11 / eps_r)
    w_over_h_A = 8 * np.exp(A) / (np.exp(2 * A) - 2)

    B = 377 * np.pi / (2 * z0 * np.sqrt(eps_r))
    w_over_h_B = (2 / np.pi) * (
        B - 1 - np.log(2 * B - 1)
        + (eps_r - 1) / (2 * eps_r) * (np.log(B - 1) + 0.39 - 0.61 / eps_r)
    )

    w_over_h = w_over_h_A if w_over_h_A < 2 else w_over_h_B
    return w_over_h * h


def build_geometry(design: DesignParams, non_graphical=False, create_port=True, create_airbox=True, use_inset=True):
    L, W, h, eps_r, y0 = design.L, design.W, design.h, design.eps_r, design.y0

    wf = _feed_line_width(h, eps_r, z0=50.0)
    feed_length = 4 * h + L / 4

    margin = 6 * h
    sub_W = W + 2 * margin
    sub_L = L + feed_length + 2 * margin

    hfss = Hfss(
        specified_version="2022.1",
        non_graphical=non_graphical,
        new_desktop_session=True,
        close_on_exit=False,
        designname="PatchAntenna",
        solution_type="DrivenModal",
    )
    hfss.modeler.model_units = "meter"

    mat_name = "patch_substrate"
    if mat_name in hfss.materials.material_keys:
        hfss.materials.remove_material(mat_name)
    mat = hfss.materials.add_material(mat_name)
    mat.permittivity = eps_r

    hfss.modeler.create_rectangle(
        csPlane="XY", position=[-sub_L / 2, -sub_W / 2, 0],
        dimension_list=[sub_L, sub_W], name="ground", matname="copper",
    )

    hfss.modeler.create_box(
        position=[-sub_L / 2, -sub_W / 2, 0],
        dimensions_list=[sub_L, sub_W, h], name="substrate", matname=mat_name,
    )

    patch_x0 = -L / 2
    patch = hfss.modeler.create_rectangle(
        csPlane="XY", position=[patch_x0, -W / 2, h],
        dimension_list=[L, W], name="patch", matname="copper",
    )

    if use_inset:
        slot_width = 2 * wf
        notch1 = hfss.modeler.create_rectangle(
            csPlane="XY", position=[patch_x0, wf / 2, h],
            dimension_list=[y0, slot_width], name="notch_pos", matname="vacuum",
        )
        notch2 = hfss.modeler.create_rectangle(
            csPlane="XY", position=[patch_x0, -wf / 2 - slot_width, h],
            dimension_list=[y0, slot_width], name="notch_neg", matname="vacuum",
        )
        hfss.modeler.subtract([patch], [notch1, notch2], keep_originals=False)

    feed_x0 = -sub_L / 2
    feed_x1 = patch_x0 if not use_inset else patch_x0 + y0
    feedline = hfss.modeler.create_rectangle(
        csPlane="XY", position=[feed_x0, -wf / 2, h],
        dimension_list=[feed_x1 - feed_x0, wf], name="feedline", matname="copper",
    )
    hfss.modeler.unite(["patch", "feedline"])
    hfss.assign_perfecte_to_sheets(["ground", "patch"])

    print("Assigned PerfectE boundary to ground and patch+feedline sheets.")
    print(f"Geometry built. Feed line width wf = {wf*1e3:.3f} mm, feed_length = {feed_length*1e3:.2f} mm")
    print(f"Substrate footprint: {sub_L*1e3:.2f} x {sub_W*1e3:.2f} mm")

    if create_port:
        port_sheet = hfss.modeler.create_rectangle(
            csPlane="YZ", position=[feed_x0, -wf / 2, 0],
            dimension_list=[wf, h], name="port_cap",
        )
        integration_line = [[feed_x0, 0, 0], [feed_x0, 0, h]]
        hfss.create_lumped_port_to_sheet(
            port_sheet.name, axisdir=integration_line, impedance=50,
            portname="P1", reference_object_list=["ground"],
        )
        print("Lumped port 'P1' created.")

    fr_est, _ = _resonant_frequency(L, W, h, eps_r)

    if create_airbox:
        lambda_est = C0 / fr_est
        pad = lambda_est / 4

        air_x0 = -sub_L / 2 - pad
        air_y0 = -sub_W / 2 - pad
        air_z0 = -pad * 0.1
        air_L = sub_L + 2 * pad
        air_W = sub_W + 2 * pad
        air_H = h + pad + pad * 0.1

        hfss.modeler.create_box(
            position=[air_x0, air_y0, air_z0],
            dimensions_list=[air_L, air_W, air_H], name="airbox", matname="vacuum",
        )
        hfss.assign_radiation_boundary_to_objects("airbox")

        print(f"Airbox created, pad = lambda/4 at fr_est={fr_est/1e9:.3f} GHz = {pad*1e3:.2f} mm")

    return hfss, fr_est


def setup_and_solve(hfss, fr_est, sweep_frac=0.15, n_points=81,
                     freq_start_hz=None, freq_stop_hz=None,
                     setup_name="Setup1", sweep_name="Sweep1"):
    if freq_start_hz is not None and freq_stop_hz is not None:
        f_lo = freq_start_hz
        f_hi = freq_stop_hz
    else:
        f_lo = fr_est * (1 - sweep_frac)
        f_hi = fr_est * (1 + sweep_frac)

    setup = hfss.create_setup(setupname=setup_name)
    setup.props["Frequency"] = f"{fr_est/1e9}GHz"
    setup.props["MaximumPasses"] = 10
    setup.props["MinimumConvergedPasses"] = 2
    setup.props["MaxDeltaS"] = 0.02
    setup.update()

    hfss.create_linear_count_sweep(
        setupname=setup_name, unit="GHz",
        freqstart=f_lo / 1e9, freqstop=f_hi / 1e9,
        num_of_freq_points=n_points, sweepname=sweep_name,
        sweep_type="Discrete", save_fields=True,
    )
    print(f"Setup '{setup_name}' created: adaptive freq {fr_est/1e9:.3f} GHz, max 10 passes, MaxDeltaS 0.02.")
    print(f"Sweep '{sweep_name}': {f_lo/1e9:.3f}-{f_hi/1e9:.3f} GHz, {n_points} points, discrete, fields saved.")
    print("Starting analysis (this may take a few minutes)...")
    hfss.analyze_setup(setup_name)
    print("Analysis complete.")
def extract_s11(hfss, setup_name="Setup1", sweep_name="Sweep1"):
    """Pull S11 (dB) vs frequency (Hz) from the completed sweep."""
    solution = hfss.post.get_solution_data(
        expressions="dB(S(1,1))",
        setup_sweep_name=f"{setup_name} : {sweep_name}",
    )
    freq_ghz = np.array(solution.primary_sweep_values, dtype=float)
    s11_db = np.array(solution.data_real(), dtype=float)
    freq_hz = freq_ghz * 1e9
    return freq_hz, s11_db

def extract_zin(hfss, setup_name="Setup1", sweep_name="Sweep1", z0=50.0):
    """
    Derive complex port input impedance Zin(f) from complex S11, since
    Driven Modal solutions only expose S-parameters as report quantities
    (Zin(...) is only available in Driven Terminal mode).
    Zin = Z0 * (1 + S11) / (1 - S11)
    """
    sol_re = hfss.post.get_solution_data(
        expressions="re(S(1,1))",
        setup_sweep_name=f"{setup_name} : {sweep_name}",
    )
    sol_im = hfss.post.get_solution_data(
        expressions="im(S(1,1))",
        setup_sweep_name=f"{setup_name} : {sweep_name}",
    )
    freq_ghz = np.array(sol_re.primary_sweep_values, dtype=float)
    re_s11 = np.array(sol_re.data_real(), dtype=float)
    im_s11 = np.array(sol_im.data_real(), dtype=float)
    gamma = re_s11 + 1j * im_s11
    zin = z0 * (1 + gamma) / (1 - gamma)
    freq_hz = freq_ghz * 1e9
    return freq_hz, zin.real, zin.imag

def add_farfield_sphere(hfss, name="Boresight"):
    """
    Create an infinite sphere at boresight (theta=0, phi=0), which for this
    geometry (patch in XY plane, radiating +Z) is the main lobe direction.
    """
    hfss.insert_infinite_sphere(
        x_start=0, x_stop=0, x_step=1,
        y_start=0, y_stop=0, y_step=1,
        name=name,
    )
    print(f"Infinite sphere '{name}' created at boresight (theta=0, phi=0).")
    return name


def extract_gain(hfss, setup_name="Setup1", sweep_name="Sweep1", sphere_name="Boresight", expression="RealizedGainTotal"):
    """Pull realized gain (dBi) at boresight vs frequency from the far-field solution."""
    solution = hfss.post.get_far_field_data(
        expression=expression,
        setup_sweep_name=f"{setup_name} : {sweep_name}",
        domain=sphere_name,
        families_dict={"Theta": ["0"], "Phi": ["0"], "Freq": ["All"]},
    )
    freq_ghz = np.array(solution.primary_sweep_values, dtype=float)
    gain_db = np.array(solution.data_real(), dtype=float)
    freq_hz = freq_ghz * 1e9
    return freq_hz, gain_db
def simulate_hfss(design: DesignParams, freq_hz: np.ndarray, non_graphical=True, n_points=81) -> SimResult:


    f_lo = float(np.min(freq_hz)) * 0.95
    f_hi = float(np.max(freq_hz)) * 1.05

    hfss, fr_est = build_geometry(
        design, non_graphical=non_graphical,
        create_port=True, create_airbox=True, use_inset=True,
    )
    try:
        add_farfield_sphere(hfss)
        setup_and_solve(hfss, fr_est, freq_start_hz=f_lo, freq_stop_hz=f_hi, n_points=n_points)

        hfss_freq_hz, s11_db_raw = extract_s11(hfss)
        _, gain_db_raw = extract_gain(hfss)

        min_idx = int(np.argmin(s11_db_raw))
        resonant_freq_hz = float(hfss_freq_hz[min_idx])
        min_s11_db = float(s11_db_raw[min_idx])
        peak_gain_dbi = float(np.max(gain_db_raw))

        below_10 = s11_db_raw <= -10.0
        if np.any(below_10):
            idx = np.where(below_10)[0]
            bandwidth_hz = float(hfss_freq_hz[idx[-1]] - hfss_freq_hz[idx[0]])
        else:
            bandwidth_hz = 0.0

        s11_db = np.interp(freq_hz, hfss_freq_hz, s11_db_raw)
        gain_dbi = np.interp(freq_hz, hfss_freq_hz, gain_db_raw)

        return SimResult(
            freq_hz=freq_hz,
            s11_db=s11_db,
            gain_dbi=gain_dbi,
            resonant_freq_hz=resonant_freq_hz,
            bandwidth_hz=bandwidth_hz,
            min_s11_db=min_s11_db,
            peak_gain_dbi=peak_gain_dbi,
        )
    finally:
        hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    test_design = DesignParams(
        L=0.02940, W=0.03800, h=0.0016, eps_r=4.40, y0=0.00750,
    )
    hfss, fr_est = build_geometry(test_design, non_graphical=False, create_port=True, create_airbox=True,
                                  use_inset=True)
    add_farfield_sphere(hfss)
    setup_and_solve(hfss, fr_est)


    freq_hz, s11_db = extract_s11(hfss)
    min_idx = np.argmin(s11_db)
    print(f"\n=== HFSS S11 results ===")
    print(f"Resonant freq (min S11): {freq_hz[min_idx]/1e9:.3f} GHz")
    print(f"Min S11: {s11_db[min_idx]:.2f} dB")
    print("Full S11 curve (GHz, dB):")
    for f, s in zip(freq_hz, s11_db):
        print(f"  {f/1e9:.3f}  {s:.2f}")
        freq_hz_z, re_zin, im_zin = extract_zin(hfss)
        print(f"\n=== HFSS Zin results ===")
        print("Full Zin curve (GHz, Re(Zin) ohm, Im(Zin) ohm):")
        for f, r, im in zip(freq_hz_z, re_zin, im_zin):
            print(f"  {f / 1e9:.3f}  {r:.2f}  {im:.2f}")
            freq_hz_g, gain_db = extract_gain(hfss)
            print(f"\n=== HFSS Gain results (boresight) ===")
            peak_idx = np.argmax(gain_db)
            print(f"Peak realized gain: {gain_db[peak_idx]:.2f} dBi at {freq_hz_g[peak_idx] / 1e9:.3f} GHz")
            print("Full gain curve (GHz, dBi):")
            for f, g in zip(freq_hz_g, gain_db):
                print(f"  {f / 1e9:.3f}  {g:.2f}")

    input("Press Enter to close AEDT...")
    hfss.release_desktop(close_projects=True, close_desktop=True)