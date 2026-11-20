"""Contact sheet of randomly sampled IU picks for the manual audit of the frontal selection.
python tools/iu_sheet.py manifests/iu.csv iu_audit.jpg 120"""
import sys
import pandas as pd
from PIL import Image, ImageDraw

n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
df = pd.read_csv(sys.argv[1]).sample(n=n, random_state=1).reset_index(drop=True)
T, cols = 150, 12
rows = -(-n // cols)
sheet = Image.new("L", (cols * T, rows * (T + 12)), 255); d = ImageDraw.Draw(sheet)
for k, r in enumerate(df.itertuples()):
    im = Image.open(r.image_path).convert("L"); im.thumbnail((T, T))
    x, y = (k % cols) * T, (k // cols) * (T + 12)
    sheet.paste(im, (x, y)); d.text((x + 2, y + T), f"#{k} p={r.p_frontal:.2f}", fill=0)
sheet.save(sys.argv[2], quality=88); print("wrote", sys.argv[2], f"({n} images, numbered #0..#{n-1} left to right, top to bottom)")
