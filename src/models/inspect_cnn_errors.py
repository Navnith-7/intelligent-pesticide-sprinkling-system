
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "results" / "cnn_test_predictions.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "corn_disease_errors.png"
)

df = pd.read_csv(PREDICTIONS_PATH)

# Focus on the main confusion identified by the error analysis.
actual = "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
predicted = "Corn_(maize)___Northern_Leaf_Blight"

errors = df[
    (df["true_label"] == actual)
    & (df["predicted_label"] == predicted)
].head(12)

if errors.empty:
    raise RuntimeError("No matching corn disease errors were found.")

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, (_, row) in zip(axes, errors.iterrows()):
    image_path = PROJECT_ROOT / row["image_path"]

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    ax.set_title(
        f"Predicted: Northern Leaf Blight\n"
        f"Confidence: {row['confidence']:.1%}",
        fontsize=9,
    )
    ax.axis("off")

fig.suptitle(
    "Test errors: Actual Gray Leaf Spot predicted as Northern Leaf Blight",
    fontsize=13,
)
plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Images inspected: {len(errors)}")
print(f"Saved visual inspection grid: {OUTPUT_PATH}")
