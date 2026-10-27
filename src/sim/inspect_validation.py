import json

d = json.load(open("data/m4_validation_results.json"))
print(f"{len(d)} designs\n")
for e in d:
    t = e["target_freq_ghz"]
    a = e["analytical"]
    h = e["hfss"]
    print(f"{t:4.1f} GHz | analytical fr={a['resonant_freq_hz']/1e9:.3f} S11={a['min_s11_db']:7.2f} gain={a['peak_gain_dbi']:5.2f} | "
          f"HFSS fr={h['resonant_freq_hz']/1e9:.3f} S11={h['min_s11_db']:7.2f} gain={h['peak_gain_dbi']:5.2f}")