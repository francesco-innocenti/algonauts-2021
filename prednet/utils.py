import os
from prednet import PredNet
from keras import Input, Model


def load_prednet(model_dir, output_mode='error', nt=16):

    # load model and weights
    weights_file = os.path.join(model_dir, 'tensorflow_weights/prednet_kitti_weights.hdf5')
    json_file = os.path.join(model_dir, 'prednet_kitti_model.json')

    with open(json_file, 'r') as f:
        json_string = f.read()
        print(json_string)

    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    train_model.load_weights(weights_file)

    # set time steps/frames
    nt = nt

    # configure model
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = output_mode
    data_format = layer_config['data_format'] if 'data_format' in layer_config \
        else layer_config['dim_ordering']
    prednet = PredNet(weights=train_model.layers[1].get_weights(),
                      **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = keras.Input(shape=tuple(input_shape))
    predictions = prednet(inputs)
    model = Model(inputs=inputs, outputs=predictions)

    return model
