# VAE-NAS
## Overview:
This is the code for “genetically” evolving a population of Variational Autoencoders. The basic model for the VAE takes input/output of 88 and one middle layer before the latent dimension. The size of the latent dimension can change as the genetic algorithm runs. The objective function currently aims to minimize loss, as calculated in the train function. The forward pass of the VAE uses the reparametrize trick to ensure the latent dimension samples differentiable values.

The dataset the VAE’s are trained on are the AMASS body motion dataset. I have tested it on C3 - run_poses.npz in particular. In smpl_data.py, I extract the joint data and reshape it to be a 3D array of shape (326, 22, 4). Then I reshape that quaternion data to be of shape (326, 88), where each sample contains the quaternions for all the joints (all in same row)
