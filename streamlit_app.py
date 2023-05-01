
import streamlit as st
import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage import measure


def count_colonies(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to the grayscale image
    T = threshold_local(gray, 21, offset = 10, method = "gaussian")
    binary = (gray > T).astype("uint8") * 255
    
    # Find contours of the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw masks on top of the identified contours
    masks = np.zeros_like(gray)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 50 and area < 500:
            cv2.drawContours(masks, contours, i, (255, 255, 255), -1)
    
    # Count the number of identified colonies
    labels = measure.label(masks)
    num_colonies = len(np.unique(labels)) - 1
    
    # Overlay masks on top of the original image
    overlay = cv2.bitwise_and(image, image, mask = masks)
    
    return num_colonies, overlay


def main():
    # Set title of the application
    st.title("Bacteria Colony Counter")
    
    # Allow user to upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        
        # Call the 'count_colonies' function
        num_colonies, overlay = count_colonies(image)
        
        # Display the original image and the overlay
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(overlay, caption="Overlay ({} colonies)".format(num_colonies), use_column_width=True)


if __name__ == '__main__':
    main()


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


