from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load and OCR the image
image_path = './sampleImages/7.png'  # Replace with your actual image path
image = Image.open(image_path).convert('RGB')
ocr_result = ocr.ocr(image_path, cls=True)

# Build metadata list from OCR results
metadata_list = []
for line in ocr_result[0]:
    box = line[0]                   # 4-point polygon
    text = line[1][0].strip()
    score = line[1][1]

    if not text:  # Skip empty strings
        continue

    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    metadata_list.append({
        'text': text,
        'box': box,                 # Keep the original box
        'x': x_min,                 # Top-left x
        'y': y_min,                 # Top-left y
        'w': x_max - x_min,
        'h': y_max - y_min,
        'center_y': (box[0][1] + box[3][1]) / 2,
        'start_x': min(box[0][0], box[3][0]),
        'score': score
    })

# Debug print
print("\nMetadata List:")
for data in metadata_list:
    print("\t", data)

# Sort lines by Y-center (vertical position)
lines_sorted = sorted(metadata_list, key=lambda x: x['center_y'])

# Compute vertical gap threshold
y_centers = [line['center_y'] for line in lines_sorted]
vertical_gaps = [y2 - y1 for y1, y2 in zip(y_centers[:-1], y_centers[1:])]
median_gap = np.median(vertical_gaps) if vertical_gaps else 0
vertical_thresh = median_gap * 1.3  # allow 30% more spacing within paragraphs

# Compute indent threshold using IQR
start_xs = [line['start_x'] for line in lines_sorted]
q1, q3 = np.percentile(start_xs, [25, 75]) if len(start_xs) >= 4 else (min(start_xs), max(start_xs))
iqr = q3 - q1
indent_thresh = iqr * 1.5  # lines with start_x > this compared to previous are considered indents

# Group into paragraphs
paragraphs = []
current_paragraph = [lines_sorted[0]]
prev_y = lines_sorted[0]['center_y']
prev_x = lines_sorted[0]['start_x']

for i in range(1, len(lines_sorted)):
    curr = lines_sorted[i]
    curr_y = curr['center_y']
    curr_x = curr['start_x']

    is_new_para = abs(curr_y - prev_y) > vertical_thresh or (curr_x - prev_x) > indent_thresh
    if is_new_para:
        paragraphs.append(current_paragraph)
        current_paragraph = [curr]
    else:
        current_paragraph.append(curr)

    prev_y, prev_x = curr_y, curr_x

paragraphs.append(current_paragraph)  # Don't forget last one

# Merge bounding boxes for paragraphs
def merge_boxes(meta_list):
    all_pts = [pt for meta in meta_list for pt in meta['box']]
    xs = [pt[0] for pt in all_pts]
    ys = [pt[1] for pt in all_pts]
    return [[min(xs), min(ys)], [max(xs), min(ys)],
            [max(xs), max(ys)], [min(xs), max(ys)]]

paragraph_boxes = [merge_boxes(p) for p in paragraphs]

# Draw paragraph bounding boxes on image
draw = ImageDraw.Draw(image)
for box in paragraph_boxes:
    draw.polygon([tuple(pt) for pt in box], outline='blue', width=2)

# Show the image with paragraph boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title("Detected Paragraphs")
plt.show()