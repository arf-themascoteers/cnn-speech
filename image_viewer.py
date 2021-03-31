import matplotlib.pyplot as plt


def view_image(image):
    plt.imshow((image.reshape(28, 28)))  # Reshape as image is a vector already
    plt.show()


