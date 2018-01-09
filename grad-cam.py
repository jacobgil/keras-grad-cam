from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import sys
import cv2


def build_model():
    """Function returning keras model instance.

    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    return VGG16(weights='imagenet')


def build_guided_model():
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if 'GuidedBackProp' not in ops._gradient_registry._registry:
        @ops.RegisterGradient('GuidedBackProp')
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def load_image(path, preprocess=True):
    """Function to load and preprocess image."""
    x = image.load_img(path, target_size=(224, 224))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def guided_backprop(model, img, activation_layer):
    """Compute gradients of conv. activation w.r.t. the input image.

    If model is modified properly this will result in Guided Backpropagation
    method for visualizing input saliency.
    See https://arxiv.org/abs/1412.6806 """
    input_img = model.input
    layer_output = model.get_layer(activation_layer).output
    grads = K.gradients(layer_output, input_img)[0]
    gradient_fn = K.function([input_img, K.learning_phase()], [grads])
    grads_val = gradient_fn([img, 0])[0]
    return grads_val


def grad_cam(input_model, img, category_index, activation_layer):
    """GradCAM method for visualizing input saliency."""
    loss = input_model.output[0, category_index]
    layer_output = input_model.get_layer(activation_layer).output
    grads = normalize(K.gradients(loss, layer_output)[0])
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([img, 0])
    conv_output, grads_val = conv_output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(conv_output, weights)

    cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_saliency(model, guided_model, layer_name, img_path, cls=-1, visualize=True, save=True):
    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
    top_n = 5
    top = decode_predictions(predictions, top=top_n)[0]
    classes = np.argsort(predictions[0])[-top_n:][::-1]
    print('Model prediction:')
    for c, p in zip(classes, top):
        print('\t({}) {:20s}\twith probability {:.3f}'.format(c, p[1],p[2]))
    if cls == -1:
        cls = np.argmax(predictions)
    nb_classes = 1000
    class_name = decode_predictions(np.eye(1, nb_classes, cls))[0][0][1]
    print("Computing saliency for '{}'".format(class_name))

    gradcam = grad_cam(model, preprocessed_input, cls, activation_layer=layer_name)
    gb = guided_backprop(guided_model, img=preprocessed_input, activation_layer=layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + np.float32(cv2.imread(sys.argv[1]))) / 2
        cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))

    if visualize:
        plt.figure(figsize=(15, 6))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))

        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()

    return gradcam, gb, grad_cam


if __name__ == '__main__':
    model = build_model()
    guided_model = build_guided_model()
    gradcam, gb, grad_cam = compute_saliency(model, guided_model, layer_name='block5_conv3',
                                             img_path=sys.argv[1], cls=-1)
