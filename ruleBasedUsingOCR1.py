from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. Load OCR and Image
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
img_path = './sampleImages/5.png'
image = Image.open(img_path).convert('RGB')
results = ocr.ocr(img_path, cls=True)[0]

# 2. Extract OCR metadata
metadata = []
for box, (text, score) in results:
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
        'box': box
    })

# 3. Sort top to bottom
metadata.sort(key=lambda m: m['y'])

# 4. Detect columns
column_threshold = 50
columns = []

for line in metadata:
    added = False
    for col in columns:
        # Check if line x overlaps with column x
        if abs(line['x'] - col['x']) < column_threshold:
            col['lines'].append(line)
            col['x'] = (col['x'] + line['x']) / 2  # adjust centroid
            added = True
            break
    if not added:
        columns.append({'x': line['x'], 'lines': [line]})

# 5. Group into paragraphs per column
paragraphs = []
line_spacing_thresh = 15

for col in columns:
    col['lines'].sort(key=lambda m: m['y'])
    para = []
    prev_line = None
    for line in col['lines']:
        if not prev_line:
            para = [line]
        else:
            vertical_gap = line['y'] - (prev_line['y'] + prev_line['h'])
            if vertical_gap > line_spacing_thresh:
                paragraphs.append(para)
                para = [line]
            else:
                para.append(line)
        prev_line = line
    if para:
        paragraphs.append(para)

# 6. Draw paragraph boxes
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
for para in paragraphs:
    x_vals = [line['x'] for line in para]
    y_vals = [line['y'] for line in para]
    x_ends = [line['x'] + line['w'] for line in para]
    y_ends = [line['y'] + line['h'] for line in para]
    x_min, y_min = int(min(x_vals)), int(min(y_vals))
    x_max, y_max = int(max(x_ends)), int(max(y_ends))
    cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# 7. Show result
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 10))
plt.imshow(image_rgb)
plt.title("Paragraph Boxes (Rule-Based)")
plt.axis('off')
plt.show()