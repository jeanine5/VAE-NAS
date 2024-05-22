# VAE-NAS
## Overview:
This is the code for “genetically” evolving a population of Variational Autoencoders. The basic model for the VAE takes input/output of 88 and one middle layer before the latent dimension. The size of the latent dimension can change as the genetic algorithm runs. The objective function currently aims to minimize loss, as calculated in the train function. The forward pass of the VAE uses the reparametrize trick to ensure the latent dimension samples differentiable values.

The dataset the VAE’s are trained on are the AMASS body motion dataset. I have tested it on C3 - run_poses.npz in particular. In [smpl_data.py](https://github.com/jeanine5/VAE-NAS/blob/main/smpl_data.py), I extract the joint data and reshape it to be a 3D array of shape (326, 22, 4). Then I reshape that quaternion data to be of shape (326, 88), where each sample contains the quaternions for all the joints (all in the same row). I store this data as a .csv file that I then turn into a dataloader so I can train the VAEs.

Some modifications were made in the evolution and genetic functions. In [genetic_functions](https://github.com/jeanine5/VAE-NAS/blob/main/genetic_functions.py), since the structure of the VAE is stricter than that of a DNN, I only “crossover” and “mutate” the parameters of the VAEs. In [nsga.py](https://github.com/jeanine5/VAE-NAS/blob/main/nsga.py), I mix computation and prediction during the main for loop. I train the initial population and the offspring, and store the values ('latent_dim', 'mid_layer', 'loss', 'OOD') to a different csv file. Then, in the main loop when the offspring population is generated again, I predict their values. Based on the predicted values, the best N//2 VAEs are trained to compute their actual values.

##Future:
Out-of-Distribution 
I still need to implement the metric for calculating how well the VAE handles out-of-distribution samples (OOD). Some metrics I was looking at was log (or negative log) likelihood and log likelihood regret
N//2 characteristic
I need some characteristic to define the “best” VAEs based on their predicted values. Since I am doing multi-objective optimization, I wanted to use some characteristic outside one 
VAE size
This is a basic model of VAE, so input/output of 88 must be changed. Make it so that user can input desired size
