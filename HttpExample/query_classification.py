import sys              # Import for time
import os               # Import for reading files
# from gtts import gTTS   # Import Google Text to Speech

# Disable tensorflow compilation warnings
import tensorflow as tf # Import tensorflow for Inception Net's backend
import tensorboard as tb
import tempfile
from os import listdir
import logging
#---


def getLabel(image_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    # Language used by Google Text to Speech
    language = 'en'

    # Image to be classified
    # image_path = sys.argv[1]
    tempFilePath = tempfile.gettempdir()
    image_path = tempFilePath + "/H_test.jpg"
    logging.info(image_path)
    # Read the image data in tensorflow readable format
    image_data =tf.compat.v1.gfile.FastGFile(image_path, 'rb').read()

    # Load training labels file
    label_lines = [line.rstrip() for line in tf.compat.v1.gfile.GFile(tempFilePath + "/training_set_labels.txt")]

    # Load trained model's graph
    with tf.compat.v1.gfile.FastGFile(tempFilePath + "/trained_model_graph.pb", 'rb') as f:
        # Define a tensorflow graph
        graph_def = tf.compat.v1.GraphDef()
        # Read and import line by line from the trained model's graph
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # New tensorflow session for classification
    with tf.compat.v1.Session() as sess:

        # Feed the image data to the graph of the trained model and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # Get prediction by decoding the jpg image
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort the predictions in descending order based on score
        sorted_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

        # Print the predicted letter and the score
        print("\n\nPredicted Letter: ", str(label_lines[sorted_predictions[0]]).upper(), "\tScore: ", predictions[0][sorted_predictions[0]], "\n\n")

        # Create the text to be spoken
        prediction_text = "The predicted letter is " + str(label_lines[sorted_predictions[0]])
    return prediction_text
    # Create a speech object from text to be spoken
#     speech_object = gTTS(text=prediction_text, lang=language, slow=False)

    # Save the speech object in a file called 'prediction.mp3'
#     speech_object.save("prediction.mp3")
 
    # Playing the speech using mpg321
#     os.system("mpg321 prediction.mp3")

    # Display the letters and the score for each prediction
    '''
    for letter_prediction in sorted_predictions:
        letter = label_lines[letter_prediction]
        score = predictions[0][letter_prediction]
        print('%s (score = %.5f)' % (letter, score))'''