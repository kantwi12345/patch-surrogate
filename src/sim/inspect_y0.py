import json

d = json.load(open("data/m5_y0_sweep.json"))
print(f"{len(d)} points total")
for name in sorted(set(r["name"] for r in d)):
    print(f"\n--- {name} ---")
    for r in d:
        if r["name"] == name:
            print(f"  y0_frac={r['y0_frac']:.2f}  Rin={r['Rin']:7.2f}  Xin={r['Xin']:7.2f}")