from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load and OCR the image
image_path = './sampleImages/9.png'  # Update with your image path
image = Image.open(image_path).convert('RGB')
ocr_result = ocr.ocr(image_path, cls=True)

# Build metadata list from OCR results
metadata_list = []
for line in ocr_result[0]:
    box = line[0]
    text = line[1][0].strip()
    score = line[1][1]

    if not text:
        continue

    x_coords = [pt[0] for pt in box]
    y_coords = [pt[1] for pt in box]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    center_x = np.mean([pt[0] for pt in box])
    center_y = np.mean([pt[1] for pt in box])
    start_x = min(box[0][0], box[3][0])

    metadata_list.append({
        'text': text,
        'box': box,
        'x': x_min,
        'y': y_min,
        'w': x_max - x_min,
        'h': y_max - y_min,
        'center_x': center_x,
        'center_y': center_y,
        'start_x': start_x,
        'score': score
    })

# === 1. Detect Columns ===
# Cluster X-centers to find columns
x_centers = np.array([[meta['center_x']] for meta in metadata_list])

# You can change this to a dynamic estimation, but we'll assume 2 columns for now
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_centers)
for i, meta in enumerate(metadata_list):
    meta['column'] = int(kmeans.labels_[i])

# Sort columns from left to right
# Get average x-center for each cluster to determine order
column_order = {
    col: idx for idx, col in enumerate(
        sorted(set(kmeans.labels_), key=lambda c: np.mean([m['center_x'] for m in metadata_list if m['column'] == c]))
    )
}
for meta in metadata_list:
    meta['column_order'] = column_order[meta['column']]

# === 2. Group into paragraphs per column ===
def group_into_paragraphs(lines):
    if not lines:
        return []

    # Sort by vertical position
    lines_sorted = sorted(lines, key=lambda x: x['center_y'])

    # Estimate dynamic thresholds
    heights = [line['h'] for line in lines_sorted]
    line_height = np.median(heights)
    vertical_thresh = line_height * 5

    start_xs = [line['start_x'] for line in lines_sorted]
    if len(start_xs) >= 4:
        q1, q3 = np.percentile(start_xs, [25, 75])
        iqr = q3 - q1
        indent_thresh = iqr * 1.5
    else:
        indent_thresh = 40  # fallback if too few lines

    # Initialize grouping
    paragraphs = []
    current_paragraph = [lines_sorted[0]]
    prev = lines_sorted[0]

    for curr in lines_sorted[1:]:
        y_gap = curr['center_y'] - prev['center_y']
        x_gap = curr['start_x'] - prev['start_x']

        # Paragraph break conditions
        is_new_paragraph = (
            y_gap > vertical_thresh or
            (x_gap > indent_thresh and y_gap < vertical_thresh)
        )

        if is_new_paragraph:
            paragraphs.append(current_paragraph)
            current_paragraph = [curr]
        else:
            current_paragraph.append(curr)

        prev = curr

    paragraphs.append(current_paragraph)
    return paragraphs

# Group per column and collect all paragraph boxes
paragraph_boxes = []

for col in sorted(set(column_order.values())):
    col_lines = [m for m in metadata_list if m['column_order'] == col]
    col_paragraphs = group_into_paragraphs(col_lines)

    for para in col_paragraphs:
        all_pts = [pt for meta in para for pt in meta['box']]
        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        box = [[min(xs), min(ys)], [max(xs), min(ys)],
               [max(xs), max(ys)], [min(xs), max(ys)]]
        paragraph_boxes.append(box)

# === 3. Draw Paragraph Bounding Boxes ===
draw = ImageDraw.Draw(image)
for box in paragraph_boxes:
    draw.polygon([tuple(pt) for pt in box], outline='blue', width=2)

# Show image
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.title("Paragraph Detection with Columns")
plt.show()