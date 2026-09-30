import FinanceDataReader as fdr, json
syms = ["KS11","KQ11","005930","000660","035420","005380","051910","006400","035720","068270",
        "105560","055550","012330","028260","066570","003550","096770","034730","015760","017670"]
out = {}
for s in syms:
  try:
    df = fdr.DataReader(s, "2024-01-01")
    out[s] = [float(v) for v in df["Close"].dropna().values[-540:]]
  except Exception as e: print(s, e)
json.dump(out, open("build/series.json", "w"))
print({k: len(v) for k, v in out.items()})
