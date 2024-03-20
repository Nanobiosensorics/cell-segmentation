import matplotlib.pyplot as plt


# Shows every microscopic image and marker in a batch using matplotlib.
def show_input_batch(mic_imgs, markers):
    assert len(mic_imgs) == len(markers)
    n = len(mic_imgs)
    fig, ax = plt.subplots(nrows=n, ncols=2, squeeze=False);
    fig.suptitle("Visualization of input batch");
    for i in range(n):
        ax[i][0].axis("off")
        ax[i][0].set_title("Image");
        ax[i][0].imshow(mic_imgs[i]);
        ax[i][1].axis("off");
        ax[i][1].set_title("Marker");
        ax[i][1].imshow(markers[i], cmap="jet");


# Shows every microscopic image, markers and the corresponding predictions
def show_with_predictions(mic_imgs, markers, predictions):
    assert len(mic_imgs) == len(markers) == len(predictions)
    n = len(mic_imgs)
    fig, ax = plt.subplots(nrows=n, ncols=3, squeeze=False);
    fig.suptitle("Visualization of input and predictions");
    for i in range(n):
        ax[i][0].axis("off")
        ax[i][0].set_title("Image");
        ax[i][0].imshow(mic_imgs[i]);
        ax[i][1].axis("off");
        ax[i][1].set_title("Marker");
        ax[i][1].imshow(markers[i], cmap="jet");
        ax[i][2].axis("off");
        ax[i][2].set_title("Prediction");
        ax[i][2].imshow(predictions[i], cmap="jet");

