import ml_collections


def get_config():
    default_config = dict(
        # General parameters
        function="",
        # Specifies the function to be approximated, e.g., SineShirp
        max=0.0,
        # Specifies the maximum value on which the function will be evaluated. e.g., -15.0.
        min=0.0,
        # Specifies the maximum value on which the function will be evaluated. e.g., 0.0.
        no_samples=0,
        # Specifies the number of samples that will be taken from the selected function,
        # between the min and max values.
        padding=0,
        # Specifies an amount of zero padding steps which will be concatenated at the end of
        # the sequence created by function(min, max, no_samples).
        optim="",
        # The optimizer to be used, e.g., Adam.
        lr=0.0,
        # The lr to be used, e.g., 0.001.
        no_iterations=0,
        # The number of training iterations to be executed, e.g., 20000.
        seed=0,
        # The seed of the run. e.g., 0.
        device="",
        # The device in which the model will be deployed, e.g., cuda.
        # Parameters of ConvKernel
        kernelnet_norm_type="",
        # If model == CKCNN, the normalization type to be used in the MLPs parameterizing the convolutional
        # kernels. If kernelnet_activation_function==Sine, no normalization will be used. e.g., LayerNorm.
        kernelnet_activation_function="",
        # If model == CKCNN, the activation function used in the MLPs parameterizing the convolutional
        # kernels. e.g., Sine.
        kernelnet_no_hidden=0,
        # If model == CKCNN, the number of hidden units used in the MLPs parameterizing the convolutional
        # kernels. e.g., 32.
        # Parameters of SIREN
        kernelnet_omega_0=0.0,
        # If model == CKCNN, kernelnet_activation_function==Sine, the value of the omega_0 parameter, e.g., 30.
        comment="",
        # An additional comment to be added to the config.path parameter specifying where
        # the network parameters will be saved / loaded from.
    )
    default_config = ml_collections.ConfigDict(default_config)
    return default_config
