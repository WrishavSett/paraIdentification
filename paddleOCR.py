from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load model
img_path = './sampleImages/2.png'

# OCR
results = ocr.ocr(img_path, cls=True)
image = Image.open(img_path).convert('RGB')

# Extract box, text, score
boxes = [line[0] for line in results[0]]
lines = [line[1][0] for line in results[0]]
scores = [line[1][1] for line in results[0]]

# Step 1: Create metadata list
metadata_list = []
for box, line, score in zip(boxes, lines, scores):
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    metadata_list.append({
        'text': line,
        'x': x_min,
        'y': y_min,
        'w': x_max - x_min,
        'h': y_max - y_min
    })

# Step 2: Sort by 'y' (top of the box)
metadata_list.sort(key=lambda item: item['y'])

# Step 3: Group into paragraphs using vertical threshold
thresh = 25
paragraphs = []
current_para = []

prev_y = None
for meta in metadata_list:
    if prev_y is None:
        current_para.append(meta)
    elif abs(meta['y'] - prev_y) <= thresh:
        current_para.append(meta)
    else:
        paragraphs.append({"para": current_para})
        current_para = [meta]
    prev_y = meta['y']

# Append the last paragraph if any
if current_para:
    paragraphs.append({"para": current_para})

# Step 4: Print result
print("\n===Paragraphs Metadata===\n")
for i, para in enumerate(paragraphs, 1):
    print(f"Paragraph {i}:")
    for line in para["para"]:
        print(line)
    print()