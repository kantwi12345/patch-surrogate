"""
M4 — Real HFSS simulation via pyaedt, built incrementally.

STAGE 1: geometry only (substrate, ground, inset-fed patch). Run this with
the AEDT GUI visible and visually confirm the model looks like a correct
inset-fed microstrip patch antenna before adding port/airbox/setup/solve.
"""

import numpy as np
from pyaedt import Hfss

from src.sim.schema import DesignParams

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


def build_geometry(design: DesignParams, non_graphical=False):
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
    print("AEDT GUI should now show: ground plane, substrate block, and an inset-fed patch with a feed line.")

    return hfss


if __name__ == "__main__":
    # test design: roughly the 2.4GHz design from M3's invert.py output
    test_design = DesignParams(
        L=0.02940, W=0.03800, h=0.0016, eps_r=4.40, y0=0.01090,
    )
    hfss = build_geometry(test_design, non_graphical=False)
    input("Press Enter to close AEDT...")
    hfss.release_desktop(close_projects=True, close_desktop=True)