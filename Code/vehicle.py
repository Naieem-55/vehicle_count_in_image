import cv2
import numpy as np

def detect_vehicles(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0
    for contour in contours:
        # Filter out small contours that are not vehicles
        if cv2.contourArea(contour) < 500:
            continue

        # Draw bounding box around detected vehicles
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        vehicle_count += 1

    # Add the vehicle count text to the image
    cv2.putText(image, f'Vehicle Count: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Save the result image to the specified output path
    cv2.imwrite(output_path, image)

    # Display the result 
    cv2.imshow('Vehicle Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Number of vehicles detected : {vehicle_count}')

input_path = r'C:\Users\Hp\OneDrive\Desktop\Vehicle_Count\Input Image\car2.jpg'
output_path = r'C:\Users\Hp\OneDrive\Desktop\Vehicle_Count\Output Image/output_image.jpg'

# Example usage
detect_vehicles(input_path, output_path)
