import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ----------- Step 1: OCR Initialization -----------
ocr = PaddleOCR(use_angle_cls=True, lang='en')
image_path = './sampleImages/9.png'
image = Image.open(image_path).convert('RGB')
ocr_result = ocr.ocr(image_path, cls=True)

# ----------- Step 2: Create Metadata -----------
metadata_list = []
for line in ocr_result[0]:
    box, (text, score) = line
    x_coords = [pt[0] for pt in box]
    y_coords = [pt[1] for pt in box]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    metadata_list.append({
        'text': text,
        'x': x_min,
        'y': y_min,
        'w': x_max - x_min,
        'h': y_max - y_min,
        'box': box,
        'center_y': (box[0][1] + box[3][1]) / 2,
        'start_x': min(box[0][0], box[3][0])
    })

# ----------- Step 3: Auto Column Detection via DBSCAN on X -----------
x_starts = np.array([[m['start_x']] for m in metadata_list])
db = DBSCAN(eps=40, min_samples=2).fit(x_starts)  # eps auto-groups nearby X values
labels = db.labels_  # -1 means noise

for meta, label in zip(metadata_list, labels):
    meta['column_order'] = label

valid_metadata = [m for m in metadata_list if m['column_order'] != -1]
unique_columns = sorted(set(m['column_order'] for m in valid_metadata))

# ----------- Step 4: Group Lines into Paragraphs Per Column -----------
def group_into_paragraphs(lines):
    if not lines:
        return []

    # Sort by vertical position
    lines_sorted = sorted(lines, key=lambda x: x['center_y'])

    # Estimate dynamic thresholds
    heights = [line['h'] for line in lines_sorted]
    line_height = np.median(heights)
    vertical_thresh = line_height * 1.5

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

# ----------- Step 5: Merge All Column Paragraphs -----------
all_paragraphs = []
for col_id in unique_columns:
    col_lines = [m for m in valid_metadata if m['column_order'] == col_id]
    col_paragraphs = group_into_paragraphs(col_lines)
    all_paragraphs.extend(col_paragraphs)

# ----------- Step 6: Merge Boxes of Paragraphs -----------
def merge_boxes(paragraph):
    all_pts = [pt for line in paragraph for pt in line['box']]
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    return [[min(xs), min(ys)], [max(xs), min(ys)],
            [max(xs), max(ys)], [min(xs), max(ys)]]

paragraph_boxes = [merge_boxes(p) for p in all_paragraphs]

# ----------- Step 7: Draw -----------
draw = ImageDraw.Draw(image)
for box in paragraph_boxes:
    draw.polygon([tuple(pt) for pt in box], outline='blue', width=2)

plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.show()