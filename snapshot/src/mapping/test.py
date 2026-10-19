import pandas as pd

hyb = pd.read_csv("../data/hybrid_biu_keypoints_v2.csv")
pid = pd.read_csv("../data/frame_pid_map.csv")
pid["frame"] = pid["frame"].str.replace(r"^[^/]+/", "", regex=True)

df = (hyb.merge(pid, on="frame", how="inner")
          .sort_values(["frame", "pid"])
          .reset_index(drop=True))

print("Total rows :", len(df))
print("Unique frames :", df['frame'].nunique())
print("Unique PIDs :", df['pid'].nunique())

# How many frames have more than one PID?
frames_multi_pid = (df.groupby("frame")["pid"].nunique() > 1).sum()
print("Frames with >1 PID :", frames_multi_pid)

# How many PIDs are shorter than 10 frames?
pids_short = (df.groupby("pid")["frame"].nunique() < 10).sum()
print("PIDs with <10 frames :", pids_short)
