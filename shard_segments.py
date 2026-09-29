import os
import shutil

base = "/home/dj/Documents/ml.matugen/dataset/corpus/segments"
files = os.listdir(base)
print("total files:", len(files))
moved = 0
for fn in files:
    prefix = fn[:1]
    dst_dir = os.path.join(base, prefix)
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.join(base, fn)
    if os.path.isdir(src):
        continue
    shutil.move(src, os.path.join(dst_dir, fn))
    moved += 1
print("moved:", moved)
# report new layout
counts = {}
for d in os.listdir(base):
    p = os.path.join(base, d)
    if os.path.isdir(p):
        counts[d] = len(os.listdir(p))
print("dirs:", dict(sorted(counts.items())))
print("max per dir:", max(counts.values()) if counts else 0)