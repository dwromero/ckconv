## Execution commands to replicate our results
In this notebook, we summarize the commands given to our networks in order to get the reported results.

### Instructions

###### Pretrained
If you want to use a pretrained model, please add the key `--config.pretrained=True` to the corresponding execution line.

###### Testing
If you only want to perform testing, pleases add the key `--config.train=False` to the corresponding execution line.

### sMNIST

100k

`run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=MNIST --config.device=cuda --config.dropout=0.1 --config.dropout_in=0.1 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=31.09195739463897 --config.lr=0.001 --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau`

1M

` run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=MNIST --config.device=cuda --config.dropout=0.2 --config.dropout_in=0.2 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=30.5 --config.lr=0.001 --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=100 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0`

### pMNIST

100k

` run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=MNIST --config.device=cuda --config.dropout=0 --config.dropout_in=0.1 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=43.45667389601194 --config.lr=0.001 --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=True --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau `

1m

` run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=MNIST --config.device=cuda --config.dropout=0 --config.dropout_in=0.2 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=42.161651744476515 --config.lr=0.001 --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=100 --config.optimizer=Adam --config.permuted=True --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau `

### sCIFAR10

100k

` run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=CIFAR10 --config.device=cuda --config.dropout=0.2 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=25.696 --config.lr=0.001 --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0` 

1m

` run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=CIFAR10 --config.device=cuda --config.dropout=0.3 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=25.696 --config.lr=0.001 --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=100 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0.0001 --config.weight_dropout=0.1` 

### CharTrajectories

###### Sampling rate test
One can change the frequency of the test set by varying the key `--config.sr_test`.

###### Sampling rate train = 1

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.dropout=0.1 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=21.44961185481323 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0`

###### Sampling rate train = 1/2

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.dropout=0.1 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=21.44961185481323 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=2 --config.weight_decay=0`

###### Sampling rate train = 1/4

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.dropout=0.1 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=21.44961185481323 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=4 --config.weight_decay=0`

###### Sampling rate train = 1/8

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.dropout=0.1 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=12.589391905289371 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=8 --config.weight_decay=0`

###### Data drop percentage = 30%

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.drop_rate=30 --config.dropout=0.2 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=17.243982444546685 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0`

###### Data drop percentage = 50%

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.drop_rate=50 --config.dropout=0 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=11.997102427300288 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0.0001`

###### Data drop percentage = 70%

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=CharTrajectories --config.device=cuda --config.drop_rate=70 --config.dropout=0.1 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=4.237433943113986 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.weight_decay=0`

### SpeechCommands (Preprocessed)

`run_experiment.py --config.batch_size=64 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.dropout=0.2 --config.dropout_in=0 --config.epochs=200 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=30.90257553169801 --config.lr=0.001 --config.mfcc=True --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=15 --config.scheduler=plateau --config.weight_decay=0`

### SpeechCommands (Raw)

###### Sampling rate test
One can change the frequency of the test set by varying the key `--config.sr_test`.

###### Sampling rate train = 1

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.dropout=0 --config.dropout_in=0 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=39.450608806633795 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=1 --config.weight_decay=0.0001 --config.weight_dropout=0`

###### Sampling rate train = 1/2

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.dropout=0 --config.dropout_in=0 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=39.450608806633795 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=2 --config.weight_decay=0.0001 --config.weight_dropout=0`

###### Sampling rate train = 1/4

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.dropout=0 --config.dropout_in=0 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=39.450608806633795 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=4 --config.weight_decay=0.0001 --config.weight_dropout=0`

###### Sampling rate train = 1/8

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.dropout=0 --config.dropout_in=0 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=39.450608806633795 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=8 --config.weight_decay=0.0001 --config.weight_dropout=0`

###### Data drop percentage = 30%

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.drop_rate=30 --config.dropout=0.1 --config.dropout_in=0 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=35.655229877794035 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=1 --config.weight_decay=0.0001 --config.weight_dropout=0`

###### Data drop percentage = 50%

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.drop_rate=50 --config.dropout=0 --config.dropout_in=0 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=31.700745675276966 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=1 --config.weight_decay=0.0001 --config.weight_dropout=0`

###### Data drop percentage = 70%

`run_experiment.py --config.batch_size=32 --config.clip=0 --config.dataset=SpeechCommands --config.device=cuda --config.drop_rate=70 --config.dropout=0.1 --config.dropout_in=0.1 --config.epochs=300 --config.kernelnet_activation_function=Sine --config.kernelnet_no_hidden=32 --config.kernelnet_norm_type=LayerNorm --config.kernelnet_omega_0=23.29255834358289 --config.lr=0.001 --config.mfcc=False --config.model=CKCNN --config.no_blocks=2 --config.no_hidden=30 --config.optimizer=Adam --config.permuted=False --config.sched_decay_factor=5 --config.sched_decay_steps=(75,) --config.sched_patience=20 --config.scheduler=plateau --config.sr_train=1 --config.weight_decay=0 --config.weight_dropout=0.1`