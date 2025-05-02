from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Load OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = './sampleImages/1.png'
results = ocr.ocr(img_path, cls=True)
image = Image.open(img_path).convert('RGB')

# 2. Extract metadata
metadata = []
for box, (text, score) in results[0]:
    x_coords = [pt[0] for pt in box]
    y_coords = [pt[1] for pt in box]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    metadata.append({
        'text': text,
        'x': x_min,
        'y': y_min,
        'w': x_max - x_min,
        'h': y_max - y_min,
        'x_center': (x_min + x_max) / 2,
        'y_center': (y_min + y_max) / 2
    })

# 3. Cluster lines into paragraphs using DBSCAN
coords = np.array([[m['x_center'], m['y_center']] for m in metadata])
db = DBSCAN(eps=40, min_samples=2).fit(coords)
labels = db.labels_

# Assign cluster labels
for i, m in enumerate(metadata):
    m['cluster'] = labels[i]

# 4. Group by cluster
paragraphs = {}
for m in metadata:
    if m['cluster'] == -1:
        continue  # noise
    paragraphs.setdefault(m['cluster'], []).append(m)

# 5. Draw paragraphs
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
for lines in paragraphs.values():
    x_vals = [m['x'] for m in lines]
    y_vals = [m['y'] for m in lines]
    x_ends = [m['x'] + m['w'] for m in lines]
    y_ends = [m['y'] + m['h'] for m in lines]
    x_min, y_min = int(min(x_vals)), int(min(y_vals))
    x_max, y_max = int(max(x_ends)), int(max(y_ends))
    cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# 6. Show final result
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 10))
plt.imshow(image_rgb)
plt.title("Paragraph Grouping via Layout Clustering")
plt.axis('off')
plt.show()