import ml_collections


def get_config():
    default_config = dict(
        # --------------------------
        # General parameters
        dataset="",
        # The dataset to be used, e.g., MNIST.
        model="",
        # The model to be used, e.g., CKCNN.
        optimizer="",
        # The optimizer to be used, e.g., Adam.
        device="",
        # The device in which the model will be deployed, e.g., cuda.
        scheduler="",
        # The lr scheduler to be used, e.g., multistep, plateau.
        sched_decay_steps=(400,),
        # If scheduler == multistep, this specifies the steps at which
        # The scheduler should be decreased.
        sched_decay_factor=1.0,
        # The factor with which the lr will be reduced, e.g., 5, 10.
        sched_patience=0,
        # If scheduler == plateau, the number of steps to wait without
        # improvement on the validation loss until lr reduction.
        lr=0.0,
        # The lr to be used, e.g., 0.001.
        optimizer_momentum=0.0,  # **Not used in our experiments**.
        # If optimizer == SGD, this specifies the momentum of the SGD.
        clip=0.0,  # **Not used in CKCNNs / BFCNNs.**
        # Clips the gradient before loss.backward() to the specified value
        # if clip > 0.0
        no_hidden=0,
        # The number of channels at the hidden layers of the main network, e.g., 30.
        no_blocks=2,
        # The number of residual blocks in the network, e.g., 2.
        whitening_scale=1.0,  # **Not used in our experiments.**
        # Specifies a factor with which the current variance initialization is weighted.
        weight_decay=0.0,
        # Specifies a L2 norm over the magnitude of the weigths in the network, e.g., 1e-4.
        # **Important** If model == CKCNN, this loss is calculated over the sampled kernel
        # and not over the parameters of the MLP parameterizing the kernel.
        dropout=0.0,
        # Specifies a layer-wise dropout factor, e.g., 0.1.
        dropout_in=0.0,
        # Applies dropout over the input signal of the network, e.g., 0.1.
        weight_dropout=0.0,  # **Only used in CIFAR10.**
        # Applies dropout over the sampled convolutional kernel, e.g., 0.1.
        batch_size=0,
        # The batch size to be used, e.g., 64.
        epochs=0,
        # The number of epochs to perform training, e.g., 200.
        seed=0,
        # The seed of the run. e.g., 0.
        comment="",
        # An additional comment to be added to the config.path parameter specifying where
        # the network parameters will be saved / loaded from.
        pretrained=False,
        # Specifies if a pretrained model should be loaded.
        train=True,
        # Specifies if training should be performed.
        augment=False,  # **No augment used in our experiments.**
        path="",
        # This parameter is automatically derived from the other parameters of the run. It specifies
        # the path where the network parameters will be saved / loaded from.
        report_auc=False,
        max_epochs_no_improvement=100,
        # --------------------------
        # Parameters of TCNs / BFCNNs
        cnn_kernel_size=0,
        # If model in [TCN, BFCNN], the kernel size of a conventional CNN.
        # Parameters of CKCNNs
        kernelnet_norm_type="",
        # If model == CKCNN, the normalization type to be used in the MLPs parameterizing the convolutional
        # kernels. If kernelnet_activation_function==Sine, no normalization will be used. e.g., LayerNorm.
        kernelnet_activation_function="",
        # If model == CKCNN, the activation function used in the MLPs parameterizing the convolutional
        # kernels. e.g., Sine.
        kernelnet_no_hidden=0,
        # If model == CKCNN, the number of hidden units used in the MLPs parameterizing the convolutional
        # kernels. e.g., 32.
        pool=False,  # **Not used in our experiments -> Worse performance.**
        # If True, it adds a max pool layer after each Residual Block.
        # --------------------------
        # Parameters of SIREN
        kernelnet_omega_0=0.0,
        # If model == CKCNN, kernelnet_activation_function==Sine, the value of the omega_0 parameter, e.g., 30.
        # --------------------------
        # Parameters of Datasets
        # 1. Add prob.
        seq_length=0,
        # Specifies the length of the sequence both for dataset in [AddProblem, CopyMemory]. e.g., 1000.
        # 2. Copy memory
        memory_size=0,
        # Specifies the size of the sequence to memory in the CopyMemory task. e.g., 10.
        # 3. MNIST
        permuted=False,
        # Specifies if we want to use sMNIST (False) or pMNIST (True).
        # 4. SpeechCommands
        mfcc=True,
        # Specifies if we want to use the SC (True) or the SC_Raw (False).
        # 5. Continuous Datasets
        sr_train=1,
        # Specifies the factor with which the original train dataset will be downsampled. Used for experiments
        # With different training and test sampling rates. e.g., 1, 2, 4.
        sr_test=0,
        # Specifies the factor with which the original test dataset will be downsampled. Used for experiments
        # With different training and test sampling rates. e.g., 1, 2, 4.
        drop_rate=0,
        # Specifies the rate at which data will be droped from the original dataset. Used for experiments
        # With missing data. e.g., 30, 50, 70.
    )
    default_config = ml_collections.ConfigDict(default_config)
    return default_config
