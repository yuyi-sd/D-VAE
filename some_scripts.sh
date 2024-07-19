# standard training on UEs (these checkpoints can help monitor some variables and losses in our method)
CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet18                    \
                      --exp_name                experiments/cifar10/em8_train_bn_1.0 \
                      --train_data_type         PoisonCIFAR10                       \
                      --test_data_type          CIFAR10                       \
                      --train                                                 \
                      --perturb_tensor_filepath ./experiments/cifar10/em8/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise 


CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet18                    \
                      --exp_name                experiments/cifar10/lsp1_train_bn_1.0 \
                      --train_data_type         PoisonCIFAR10                       \
                      --test_data_type          CIFAR10                       \
                      --train                                                 \
                      --perturb_tensor_filepath ./experiments/cifar10/lsp1/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise 


CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet18                    \
                      --exp_name                experiments/cifar10/ops1_train_bn_1.0 \
                      --train_data_type         PoisonCIFAR10                       \
                      --test_data_type          CIFAR10                       \
                      --train                                                 \
                      --perturb_tensor_filepath ./experiments/cifar10/ops-1/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise 



# ours defense against UEs
CUDA_VISIBLE_DEVICES=0 python -u main_vae_disentangle.py    --config_path configs/cifar10     \
                      --kd                      3.0 \
                      --version                 resnet18                    \
                      --version_s               resnet18DVAE                  \
                      --exp_name                experiments/cifar10/em8_train_bn_1.0_VAE_DIS_kd3.0_Z_spatial \
                      --train_data_type         PoisonCIFAR10                       \
                      --test_data_type          CIFAR10                       \
                      --train                                                 \
                      --perturb_tensor_filepath ./experiments/cifar10/em8/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --use_y                                         \
                      --spatial_emb                                    \
                      --perturb_type            samplewise        

CUDA_VISIBLE_DEVICES=0 python -u main_vae_disentangle.py    --config_path configs/cifar10     \
                      --kd                      3.0 \
                      --version                 resnet18                    \
                      --version_s               resnet18DVAE                  \
                      --exp_name                experiments/cifar10/lsp1_train_bn_1.0_VAE_DIS_kd3.0_Z_spatial \
                      --train_data_type         PoisonCIFAR10                       \
                      --test_data_type          CIFAR10                       \
                      --train                                                 \
                      --perturb_tensor_filepath ./experiments/cifar10/lsp1/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --use_y                                         \
                      --spatial_emb                                    \
                      --perturb_type            samplewise       


CUDA_VISIBLE_DEVICES=0 python -u main_vae_disentangle.py    --config_path configs/cifar10     \
                      --kd                      3.0 \
                      --version                 resnet18                    \
                      --version_s               resnet18DVAE                  \
                      --exp_name                experiments/cifar10/ops1_train_bn_1.0_VAE_DIS_kd3.0_Z_spatial \
                      --train_data_type         PoisonCIFAR10                       \
                      --test_data_type          CIFAR10                       \
                      --train                                                 \
                      --perturb_tensor_filepath ./experiments/cifar10/ops-1/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --use_y                                         \
                      --spatial_emb                                    \
                      --perturb_type            samplewise       
