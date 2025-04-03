import matplotlib.pyplot as plt


def display_digit(img, title="Digit"):
    """
    Display an image of a digit with a title
    """
    plt.figure(figsize=(4, 4))
    plt.title(title)

    # Handle different image formats
    if len(img.shape) == 4:  # If image has batch and channel dimensions
        plt.imshow(img[0, :, :, 0], cmap=plt.cm.binary)
    elif len(img.shape) == 3:  # If image has just a channel dimension
        plt.imshow(img[:, :, 0], cmap=plt.cm.binary)
    else:  # If image is just 2D
        plt.imshow(img, cmap=plt.cm.binary)

    plt.show()


def display_prediction_results(img, prediction, digit_number, confidence):
    """
    Display the digit with prediction information
    """
    display_digit(img, f"Digit {digit_number}: Predicted as {prediction} ({confidence:.2f}%)")