def test_tensorflow_and_datasets_compatibility():
    import tensorflow as tf
    import tensorflow_datasets as tfds

    assert tf.__version__ >= '2.5.0'
    assert tfds.__version__ < '4.9.8'  # Adjust this version based on compatibility with Python 3.9

    print("TensorFlow and TensorFlow Datasets versions are compatible with Python 3.9.")