import json

d = json.load(open("data/m5_y0_bisect.json"))
print(f"{len(d)} geometries\n")
for r in d:
    print(f"{r['name']:16s}  y0_analytical={r['y0_analytical']*1e3:6.3f}mm  "
          f"y0_true={r['y0_true']*1e3:6.3f}mm  ratio={r['ratio']:.3f}  "
          f"min_s11={r['min_s11_db']:7.2f}dB")