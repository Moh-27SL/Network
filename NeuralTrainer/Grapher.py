import struct
import matplotlib.pyplot as plt

def read_trainer_data(filepath):
    with open(filepath, 'rb') as f:
        # Read number of epochs (int)
        epochs_bytes = f.read(4)
        epochs = struct.unpack('i', epochs_bytes)[0]

        # Read accuracy array (epochs doubles)
        accuracy_bytes = f.read(8 * epochs)
        accuracy = list(struct.unpack(f'{epochs}d', accuracy_bytes))

        # Read loss array (epochs doubles)
        loss_bytes = f.read(8 * epochs)
        loss = list(struct.unpack(f'{epochs}d', loss_bytes))

    return epochs, accuracy, loss

def plot_accuracy_loss(filepath):
    epochs, accuracy, loss = read_trainer_data(filepath)
    x = list(range(1, epochs + 1))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# === Usage ===
# Replace this with the path to your binary file
plot_accuracy_loss("..\\TRAINED_DUDES\\sperm_detector_multilabel_graph.bin")
