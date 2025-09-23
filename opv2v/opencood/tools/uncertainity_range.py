
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_uncertainty_colorbar(image_height=50, image_width=400, save_path="/data/s2/abhi_workspace/CoBEVT/opv2v/opencood/visualization/uncertainty_scale.png"):
    """
    Generates a horizontal uncertainty scale using OpenCV JET colormap.
    Low (left) -> High (right)
    """
    # Create a horizontal gradient [0..255]
    gradient = np.linspace(0, 255, image_width, dtype=np.uint8)
    gradient = np.tile(gradient, (image_height, 1))

    # Apply same colormap as in visualization
    heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Plot with labels
    plt.figure(figsize=(8, 2))
    plt.imshow(heatmap_rgb)
    plt.axis("off")
    plt.title("Uncertainty Scale (Low → High)", fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_uncertainty_colorbar()
