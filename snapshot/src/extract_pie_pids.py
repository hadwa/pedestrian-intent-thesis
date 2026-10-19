#!/usr/bin/env python3
"""
build_frame_true_pid_map.py  –  for PIE video-level XMLs where the
pedestrian ID is stored inside each <box> as <attribute name="id">xxx</attribute>.
"""

import os, csv, xml.etree.ElementTree as ET
from pathlib import Path

ROOT    = Path("/Users/hadwapasha/Desktop/Masters/pedestrian-intent-biu/data/PIE/annotations")
OUT_CSV = Path("frame_true_pid_map.csv")

rows, seen = [], set()

for set_folder in ROOT.iterdir():                                 # set01, set02 …
    if not set_folder.is_dir():
        continue
    for xml_path in set_folder.glob("*_annt.xml"):                # video_0000_annt.xml …
        set_id   = set_folder.name                                # set01
        video_id = xml_path.stem.replace("_annt", "")             # video_0000
        tree = ET.parse(xml_path)

        for track in tree.findall("track"):
            if track.attrib.get("label") != "pedestrian":
                continue

            # Grab the first <attribute name="id"> inside the track
            first_box = track.find("box")
            if first_box is None:
                continue
            pid_attr = first_box.find("attribute[@name='id']")
            if pid_attr is None:
                continue
            local_pid = pid_attr.text.strip()                     # e.g. 1_1_4
            true_pid  = f"{set_id}_{video_id}_pid{local_pid}"     # globally unique

            for box in track.findall("box"):
                frame_idx = box.attrib.get("frame")
                if frame_idx is None:
                    continue
                frame_name = f"{set_id}/{video_id}/frame_{int(frame_idx):06d}.jpg"
                key = (frame_name, true_pid)
                if key in seen:                                   # avoid duplicates
                    continue
                seen.add(key)
                rows.append({"frame": frame_name, "true_pid": true_pid})

print(f"Collected {len(rows):,} (frame, true_pid) pairs.")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "true_pid"])
    writer.writeheader()
    writer.writerows(rows)
print(f"✅ Saved {OUT_CSV} with {len(rows):,} rows.")
