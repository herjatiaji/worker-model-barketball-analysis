# get_corners.py
import cv2
import json
import sys

# --- CONFIGURATION ---
# Make sure your reference image is in the same folder as this script
VIDEO_FRAME_PATH = "reference_frame.jpg" 
OUTPUT_FILE = "court_corners.json"
# --- END CONFIGURATION ---

points = []

def click_event(event, x, y, flags, params):
    """Callback function to record points on mouse click."""
    global image
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        print(f"Point {len(points) + 1} selected: ({x}, {y})")
        points.append([x, y])
        cv2.circle(image, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(image, str(len(points)), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Select Corners', image)
        if len(points) == 4:
            print("\nFour points selected. Press any key to save and close.")

image = cv2.imread(VIDEO_FRAME_PATH)
if image is None:
    print(f"Error: Could not load image from {VIDEO_FRAME_PATH}")
    sys.exit()

cv2.namedWindow('Select Corners')
cv2.setMouseCallback('Select Corners', click_event)

print("IMPORTANT: Please click on the 4 corners of the court in this specific order:")
print("1. Top-Left -> 2. Top-Right -> 3. Bottom-Right -> 4. Bottom-Left")
cv2.imshow('Select Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"corners": points}, f, indent=2)
    print(f"\nCorner points saved successfully to {OUTPUT_FILE}")
else:
    print("\nDid not select 4 points. Exiting without saving.")