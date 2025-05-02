from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. Load OCR and image
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
img_path = './sampleImages/5.png'  # <- Change this to your actual image path
image = Image.open(img_path).convert('RGB')
results = ocr.ocr(img_path, cls=True)[0]

# 2. Extract line metadata
lines = []
for box, (text, score) in results:
    if score < 0.5 or not text.strip():
        continue
    x_coords = [pt[0] for pt in box]
    y_coords = [pt[1] for pt in box]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    lines.append({
        'text': text,
        'x': x_min,
        'y': y_min,
        'w': x_max - x_min,
        'h': y_max - y_min,
        'x_center': (x_min + x_max) / 2,
        'box': box
    })

# 3. Sort by top-to-bottom
lines.sort(key=lambda l: l['y'])

# 4. Detect columns using x-center clustering
column_threshold = 100  # Controls how far apart columns must be
columns = []

for line in lines:
    added = False
    for col in columns:
        if abs(line['x_center'] - col['x_center']) < column_threshold:
            col['lines'].append(line)
            col['x_center'] = np.mean([l['x_center'] for l in col['lines']])  # Update center
            added = True
            break
    if not added:
        columns.append({'x_center': line['x_center'], 'lines': [line]})


# 5. Paragraph grouping per column
def group_paragraphs(column_lines, gap_multiplier=1.2):
    column_lines.sort(key=lambda l: l['y'])
    paragraphs = []
    current_para = []

    for i, line in enumerate(column_lines):
        if not current_para:
            current_para.append(line)
            continue

        prev_line = current_para[-1]
        vertical_gap = line['y'] - (prev_line['y'] + prev_line['h'])
        avg_height = np.mean([line['h'], prev_line['h']])
        gap_threshold = gap_multiplier * avg_height

        if vertical_gap > gap_threshold:
            paragraphs.append(current_para)
            current_para = [line]
        else:
            current_para.append(line)

    if current_para:
        paragraphs.append(current_para)

    return paragraphs


# Collect all paragraphs
all_paragraphs = []
for col in columns:
    paras = group_paragraphs(col['lines'])
    all_paragraphs.extend(paras)

# 6. Draw paragraph boxes
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

for para in all_paragraphs:
    x_vals = [l['x'] for l in para]
    y_vals = [l['y'] for l in para]
    x_ends = [l['x'] + l['w'] for l in para]
    y_ends = [l['y'] + l['h'] for l in para]

    x_min, y_min = int(min(x_vals)), int(min(y_vals))
    x_max, y_max = int(max(x_ends)), int(max(y_ends))

    cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# 7. Display the result
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(16, 12))
plt.imshow(image_rgb)
plt.title("Refined Paragraph Detection")
plt.axis('off')
plt.show()