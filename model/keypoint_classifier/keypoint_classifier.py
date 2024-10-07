#!/usr/bin/env python
# -- coding: utf-8 --
#import numpy as np
#import tensorflow as tf


# class KeyPointClassifier(object):
#     def _init_(
#         self,
#         model_path='model/keypoint_classifier/model.tflite',
#         num_threads=1,
#     ):
#         self.interpreter = tf.lite.Interpreter(model_path=model_path,
#                                                num_threads=num_threads)

#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#     def _call_(
#         self,
#         landmark_list,
#     ):
#         input_details_tensor_index = self.input_details[0]['index']
#         self.interpreter.set_tensor(
#             input_details_tensor_index,
#             np.array([landmark_list], dtype=np.float32))
#         self.interpreter.invoke()

#         output_details_tensor_index = self.output_details[0]['index']

#         result = self.interpreter.get_tensor(output_details_tensor_index)

#         result_index = np.argmax(np.squeeze(result))

#         return result_index

#!/usr/bin/env python
# -- coding: utf-8 --
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(self, model_path='model/keypoint_classifier/new_model1.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def __call__(self, landmark_list):
        input_shape = self.input_details[0]['shape']  # Expected shape (1, 10, 42)
        actual_input_shape = np.array(landmark_list).shape  # Actual input shape

        # Handle input shapes
        if actual_input_shape == (42,):
            # Reshape to (1, 10, 42) by repeating the input 10 times
            landmark_list = np.tile(landmark_list, (10, 1))  # Shape becomes (10, 42)
        elif actual_input_shape == (10, 42):
            pass
        else:
            raise ValueError(f"Unexpected input shape: {actual_input_shape}. Expected (42,) or (10, 42)")

        # Reshape the input to match the expected input shape (1, 10, 42)
        input_data = np.array([landmark_list], dtype=np.float32)  # Shape becomes (1, 10, 42)
        

        # Set the tensor for the interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Retrieve output from the model
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Return the index of the maximum value in the output (classification result)
        result = np.argmax(output.squeeze())
        return result