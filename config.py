
MODEL_HYPERPARAMETERS = {
    "num_class": 91, # number of class, denoted C in paper, must include the background class (id: 0)
    "input_size": 512,
    "grid_sizes": [24], # Grid number, denoted S in the paper
    "backbone": "resnet50", # resnet50, mobilenet, mobilenetv2, xception
    "head_style": "vanilla",  # decoupled, vanilla
    "head_depth": 8,
    "fpn_channel":256
}

TRAINING_PARAMETERS = {
    "batch_size": 8,
    "num_epoch": 36,
    "steps_per_epoch": 5000,
    "learning_rates": [0.01, 0.001, 0.0001],
    "epochs": [27, 33],
    "weight_decay": 0.0001,
    "momentum": 0.9,
}


def display_config(mode):
    print()
    print("Model hyperparameters")
    print("=" * 80)
    print("Number of output class:", MODEL_HYPERPARAMETERS['num_class'])
    print("Input shape:", MODEL_HYPERPARAMETERS['num_class'], "(Current only support squared images)")
    print("Grid number(s) (S):", MODEL_HYPERPARAMETERS['grid_sizes'])
    print("Backbone network:", MODEL_HYPERPARAMETERS['backbone'])
    print("Head style:", MODEL_HYPERPARAMETERS['head_style'])
    print("Depth of head network:", MODEL_HYPERPARAMETERS['head_depth'])
    print("Number of channels of FPN network:", MODEL_HYPERPARAMETERS['fpn_channel'])
    print()

    if mode == 'train':
        print("Training parameters")
        print("=" * 80)
        print("Batch size:", TRAINING_PARAMETERS['batch_size'])
        print("Number of epochs:", TRAINING_PARAMETERS['num_epoch'])
        print("Learning rate:", TRAINING_PARAMETERS['learning_rates'])
        print("Epoch that changes the learning rate:", TRAINING_PARAMETERS['epochs'])
        print("Weigth decay:", TRAINING_PARAMETERS['weight_decay'])
        print("Momentum:", TRAINING_PARAMETERS['momentum'])
        print()
