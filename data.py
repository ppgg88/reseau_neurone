import h5py
import numpy as np
import gzip


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

def load_data_2():
    f = gzip.open('datasets/train-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 5

    import numpy as np
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    import matplotlib.pyplot as plt
    image = np.asarray(data[2]).squeeze()
    plt.imshow(image)
    plt.show()
    
    #return X_train, y_train, X_test, y_test
    
load_data_2()