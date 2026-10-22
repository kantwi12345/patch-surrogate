"""
M4 — Real HFSS simulation via pyaedt, built incrementally.

STAGE 1: geometry only (substrate, ground, inset-fed patch) - DONE, verified.
STAGE 2: lumped port at feed line board edge - DONE, verified.
STAGE 3: airbox + radiation boundary.
"""

import numpy as np
from pyaedt import Hfss

from src.sim.schema import DesignParams
from src.sim.analytical_model import _resonant_frequency

C0 = 299_792_458.0


def _feed_line_width(h, eps_r, z0=50.0):
    """
    Hammerstad synthesis: solve for microstrip trace width w (meters) that
    gives characteristic impedance z0, given substrate height h and eps_r.
    """
    A = (z0 / 60.0) * np.sqrt((eps_r + 1) / 2) + (eps_r - 1) / (eps_r + 1) * (0.23 + 0.11 / eps_r)
    w_over_h_A = 8 * np.exp(A) / (np.exp(2 * A) - 2)

    B = 377 * np.pi / (2 * z0 * np.sqrt(eps_r))
    w_over_h_B = (2 / np.pi) * (
        B - 1 - np.log(2 * B - 1)
        + (eps_r - 1) / (2 * eps_r) * (np.log(B - 1) + 0.39 - 0.61 / eps_r)
    )

    w_over_h = w_over_h_A if w_over_h_A < 2 else w_over_h_B
    return w_over_h * h


def build_geometry(design: DesignParams, non_graphical=False, create_port=True, create_airbox=True):
    L, W, h, eps_r, y0 = design.L, design.W, design.h, design.eps_r, design.y0

    wf = _feed_line_width(h, eps_r, z0=50.0)
    feed_length = 4 * h + L / 4  # enough room to route to the board edge

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

    # custom dielectric material matching eps_r
    mat_name = "patch_substrate"
    if mat_name in hfss.materials.material_keys:
        hfss.materials.remove_material(mat_name)
    mat = hfss.materials.add_material(mat_name)
    mat.permittivity = eps_r

    # ground plane at z=0
    hfss.modeler.create_rectangle(
        csPlane="XY",
        position=[-sub_L / 2, -sub_W / 2, 0],
        dimension_list=[sub_L, sub_W],
        name="ground",
        matname="copper",
    )

    # substrate from z=0 to z=h
    hfss.modeler.create_box(
        position=[-sub_L / 2, -sub_W / 2, 0],
        dimensions_list=[sub_L, sub_W, h],
        name="substrate",
        matname=mat_name,
    )

    # patch position: resonant edge-to-edge along x, feed enters from -x side
    patch_x0 = -L / 2
    patch = hfss.modeler.create_rectangle(
        csPlane="XY",
        position=[patch_x0, -W / 2, h],
        dimension_list=[L, W],
        name="patch",
        matname="copper",
    )

    # inset feed notches: cut two slots from the -x edge inward by y0,
    # leaving a conducting strip of width wf at y=0 connecting to the feed line
    slot_width = 2 * wf
    notch1 = hfss.modeler.create_rectangle(
        csPlane="XY",
        position=[patch_x0, wf / 2, h],
        dimension_list=[y0, slot_width],
        name="notch_pos",
        matname="vacuum",
    )
    notch2 = hfss.modeler.create_rectangle(
        csPlane="XY",
        position=[patch_x0, -wf / 2 - slot_width, h],
        dimension_list=[y0, slot_width],
        name="notch_neg",
        matname="vacuum",
    )
    hfss.modeler.subtract([patch], [notch1, notch2], keep_originals=False)

    # feed line: from board edge up to the inset point (x = patch_x0 + y0)
    feed_x0 = -sub_L / 2
    feed_x1 = patch_x0 + y0
    feedline = hfss.modeler.create_rectangle(
        csPlane="XY",
        position=[feed_x0, -wf / 2, h],
        dimension_list=[feed_x1 - feed_x0, wf],
        name="feedline",
        matname="copper",
    )
    hfss.modeler.unite(["patch", "feedline"])

    print(f"Geometry built. Feed line width wf = {wf*1e3:.3f} mm, feed_length = {feed_length*1e3:.2f} mm")
    print(f"Substrate footprint: {sub_L*1e3:.2f} x {sub_W*1e3:.2f} mm")

    if create_port:
        # vertical sheet at the board edge, from ground (z=0) to feed trace (z=h)
        port_sheet = hfss.modeler.create_rectangle(
            csPlane="YZ",
            position=[feed_x0, -wf / 2, 0],
            dimension_list=[wf, h],
            name="port_cap",
        )
        integration_line = [[feed_x0, 0, 0], [feed_x0, 0, h]]
        hfss.create_lumped_port_to_sheet(
            port_sheet.name,
            axisdir=integration_line,
            impedance=50,
            portname="P1",
            reference_object_list=["ground"],
        )
        print("Lumped port 'P1' created on port_cap sheet (integration line ground -> feed trace).")

    if create_airbox:
        fr_est, _ = _resonant_frequency(L, W, h, eps_r)
        lambda_est = C0 / fr_est
        pad = lambda_est / 4

        air_x0 = -sub_L / 2 - pad
        air_y0 = -sub_W / 2 - pad
        air_z0 = -pad * 0.1  # small padding below ground too, box must enclose ground plane
        air_L = sub_L + 2 * pad
        air_W = sub_W + 2 * pad
        air_H = h + pad + pad * 0.1  # from just below ground to pad above the patch

        airbox = hfss.modeler.create_box(
            position=[air_x0, air_y0, air_z0],
            dimensions_list=[air_L, air_W, air_H],
            name="airbox",
            matname="vacuum",
        )
        hfss.assign_radiation_boundary_to_objects("airbox")

        print(f"Airbox created: {air_L*1e3:.1f} x {air_W*1e3:.1f} x {air_H*1e3:.1f} mm "
              f"(pad = lambda/4 at fr_est={fr_est/1e9:.3f} GHz = {pad*1e3:.2f} mm)")
        print("Radiation boundary assigned to airbox outer faces.")

    print("AEDT GUI should now show: ground, substrate, inset-fed patch, feed line, port sheet, "
          "and a large surrounding airbox (radiation boundary, usually shown semi-transparent/wireframe).")

    return hfss


if __name__ == "__main__":
    # test design: roughly the 2.4GHz design from M3's invert.py output
    test_design = DesignParams(
        L=0.02940, W=0.03800, h=0.0016, eps_r=4.40, y0=0.01090,
    )
    hfss = build_geometry(test_design, non_graphical=False, create_port=True, create_airbox=True)
    input("Press Enter to close AEDT...")
    hfss.release_desktop(close_projects=True, close_desktop=True)