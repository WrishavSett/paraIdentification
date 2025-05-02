from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load model
img_path = './sampleImages/2.png'

# 1. Perform OCR
results = ocr.ocr(img_path, cls=True)
image = Image.open(img_path).convert('RGB')

# 2. Extract box, text, score
boxes = [line[0] for line in results[0]]
lines = [line[1][0] for line in results[0]]
scores = [line[1][1] for line in results[0]]

# 3. Create metadata list
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

print("\nMetadata List:")
for data in metadata_list:
    print("\t", data)

# 4. Sort by 'y' (top of the box)
metadata_list.sort(key=lambda item: item['y'])

# 5. Group into paragraphs using vertical threshold
thresh = 25
indent = 20

paragraphs = []
current_para = []

prev_x = None
prev_y = None

for meta in metadata_list:
    x, y = meta['x'], meta['y']

    if prev_y is None:
        current_para.append(meta)
    elif abs(y - prev_y) <= thresh:
        if (x - prev_x) > indent:
            paragraphs.append({"para": current_para})
            current_para = [meta]
        else:
            current_para.append(meta)
    else:
        # New paragraph due to vertical gap
        paragraphs.append({"para": current_para})
        current_para = [meta]

    prev_y = y
    prev_x = x

# Append the last paragraph if any
if current_para:
    paragraphs.append({"para": current_para})

# Print result
print("\nParagraphs:")
for id, paragraph in enumerate(paragraphs):
    print(f"\tParagraph {id}:")
    for idx, line in enumerate(paragraph['para']):
        print("\t\t", line)

# 6. Create Bounding Box around paragraph
import cv2
import numpy as np

# Convert PIL image to OpenCV format (BGR)
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Draw paragraph bounding boxes
for para in paragraphs:
    para_lines = para['para']
    x_vals = [line['x'] for line in para_lines]
    y_vals = [line['y'] for line in para_lines]
    x_ends = [line['x'] + line['w'] for line in para_lines]
    y_ends = [line['y'] + line['h'] for line in para_lines]

    x_min = int(min(x_vals))
    y_min = int(min(y_vals))
    x_max = int(max(x_ends))
    y_max = int(max(y_ends))

    # Draw rectangle (blue box)
    cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Convert back to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Show image with paragraph boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title("Paragraph Bounding Boxes")
plt.axis('off')
plt.show()

# # 7. Check Alignment
# import cv2
# import numpy as np
#
# # Convert PIL image to OpenCV format
# image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
# # Draw vertical lines for paragraph indentation markers
# for para in paragraphs:
#     para_lines = para['para']
#
#     x_vals = [line['x'] for line in para_lines]
#     y_vals = [line['y'] for line in para_lines]
#     y_ends = [line['y'] + line['h'] for line in para_lines]
#
#     x_line = int(min(x_vals))  # Left-most x of paragraph
#     y_top = int(min(y_vals))
#     y_bottom = int(max(y_ends))
#
#     # Draw vertical line (green)
#     cv2.line(image_cv, (x_line, y_top), (x_line, y_bottom), (0, 255, 0), 2)
#
# # Convert back to RGB and show
# image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
#
# plt.figure(figsize=(12, 10))
# plt.imshow(image_rgb)
# plt.title("Paragraph Indentation Markers")
# plt.axis('off')
# plt.show()
#
# # 8. Mark Alignment
# import cv2
# import numpy as np
#
# # Load and convert image
# image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
# # Get a base reference x-alignment (e.g., minimum x across all lines)
# alignment_x = min(meta['x'] for meta in metadata_list)
# alignment_thresh = 10  # allowable deviation in pixels
#
# # Draw the vertical reference line
# y_top = min(meta['y'] for meta in metadata_list)
# y_bottom = max(meta['y'] + meta['h'] for meta in metadata_list)
# cv2.line(
#     image_cv,
#     (int(alignment_x), int(y_top)),
#     (int(alignment_x), int(y_bottom)),
#     (0, 0, 255),
#     2)
#
# # Check and mark misaligned lines
# for meta in metadata_list:
#     x = meta['x']
#     y = meta['y']
#     h = meta['h']
#
#     if abs(x - alignment_x) > alignment_thresh:
#         # Draw a short horizontal red line from start of text box
#         cv2.line(
#             image_cv,
#             (int(x), int(y+h)),
#             (int(x + 15), int(y+h)),
#             (0, 0, 255),
#             2)
#
# # Display final result
# image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(12, 10))
# plt.imshow(image_rgb)
# plt.title("Misaligned Lines Marked in Red")
# plt.axis('off')
# plt.show()