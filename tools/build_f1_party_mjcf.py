#!/usr/bin/env python3
"""Build PartyMimic training MJCFs for the F1 29DOF from the vendor f1_serial.xml.

Two outputs, same kinematics / inertias / joint limits:

  f1_party.xml      physics only: no meshes, no actuators, no sensors; the only geoms are the
                    vendor's eight foot contact spheres (2 mm, at the sole corners, z=-0.0337).
                    This is what PartyMimic compiles and what sim2sim replays.
  f1_party_vis.xml  the same plus the vendor visual meshes (contype=conaffinity=0, group 1),
                    for rendering only.

Why: PartyMimic builds its own actuators from the EntityCfg and disables self-collision
(every colliding geom becomes contype=0, conaffinity=1), so the vendor's mesh collision hulls
and its <actuator>/<sensor>/<keyframe> blocks are both unused and expensive to compile in
MuJoCo-Warp. The foot spheres are kept because they are the ground-contact model the nine
retargeting gates were computed with.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "F1_29DOF/mjcf/robot/f1/f1_serial.xml"
OUT = REPO / "assets/f1"


def is_foot_sphere(geom) -> bool:
    if geom.get("type", "sphere") != "sphere":  # MuJoCo default type is sphere
        return False
    if geom.get("contype") in (None, "0"):
        return False
    return geom.get("size") == "0.002"


def build(keep_visual: bool) -> ET.ElementTree:
    tree = ET.parse(SRC)
    root = tree.getroot()
    for tag in ("option", "size", "sensor", "keyframe", "actuator"):
        for el in root.findall(tag):
            root.remove(el)
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            keep = is_foot_sphere(geom) or (keep_visual and geom.get("type") == "mesh")
            if not keep:
                body.remove(geom)
    if not keep_visual:
        for asset in root.findall("asset"):
            root.remove(asset)
        root.find("compiler").set("meshdir", "meshes/")
    return tree


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tree = build(False)
    tree.write(OUT / "f1_party.xml", encoding="utf-8", xml_declaration=False)
    n_geom = len(list(tree.getroot().iter("geom")))
    tree2 = build(True)
    tree2.write(OUT / "f1_party_vis.xml", encoding="utf-8", xml_declaration=False)
    print(f"f1_party.xml: {n_geom} geoms; f1_party_vis.xml: "
          f"{len(list(tree2.getroot().iter('geom')))} geoms")


if __name__ == "__main__":
    main()
