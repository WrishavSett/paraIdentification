from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load and OCR the image
image_path = './sampleImages/2.png'  # Replace with actual image path
image = Image.open(image_path).convert('RGB')
ocr_result = ocr.ocr(image_path, cls=True)

# Extract valid line boxes
line_boxes = [line[0] for line in ocr_result[0] if line[1][0].strip()]

# Get Y-center and start-X for each line
def get_center_y(box): return (box[0][1] + box[3][1]) / 2
def get_start_x(box): return min(box[0][0], box[3][0])

lines_with_meta = [(box, get_center_y(box), get_start_x(box)) for box in line_boxes]
lines_sorted = sorted(lines_with_meta, key=lambda x: x[1])  # sort by Y center

# Compute vertical gaps and dynamic threshold
y_centers = [line[1] for line in lines_sorted]
vertical_gaps = [y2 - y1 for y1, y2 in zip(y_centers[:-1], y_centers[1:])]
median_gap = np.median(vertical_gaps)
vertical_thresh = median_gap * 1.3  # allow 30% extra spacing within paragraphs

# Compute dynamic indent threshold using IQR
start_xs = [line[2] for line in lines_sorted]
q1, q3 = np.percentile(start_xs, [25, 75])
iqr = q3 - q1
indent_thresh = iqr * 1.5  # anything > 1.5*IQR rightward is considered an indent

# Group lines into paragraphs
paragraphs = []
current_paragraph = [lines_sorted[0][0]]
prev_y = lines_sorted[0][1]
prev_x = lines_sorted[0][2]

for i in range(1, len(lines_sorted)):
    box, curr_y, curr_x = lines_sorted[i]
    if abs(curr_y - prev_y) > vertical_thresh or (curr_x - prev_x) > indent_thresh:
        paragraphs.append(current_paragraph)
        current_paragraph = [box]
    else:
        current_paragraph.append(box)
    prev_y, prev_x = curr_y, curr_x

paragraphs.append(current_paragraph)

# Merge paragraph boxes
def merge_boxes(boxes):
    xs = [pt[0] for box in boxes for pt in box]
    ys = [pt[1] for box in boxes for pt in box]
    return [[min(xs), min(ys)], [max(xs), min(ys)],
            [max(xs), max(ys)], [min(xs), max(ys)]]

paragraph_boxes = [merge_boxes(p) for p in paragraphs]

# Draw paragraph bounding boxes
draw = ImageDraw.Draw(image)
for box in paragraph_boxes:
    draw.polygon([tuple(pt) for pt in box], outline='blue', width=1)

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()