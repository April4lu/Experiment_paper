#=Experiment_paper
  The experiment could verify the effect of attack mentioned by CCS'17 , which is divided into part1 and part2.


#_Experiment_part1
  The part1 is a simple 3-layer neural network with SGD. All information of net will be saved in the file named nn, which could print by nn_print.py


#_Experiment_part1_add_noise
  In this part, some Gaussian noise was added to gardients which lead to a lower accuracy.


#_Experiment_part2
  The part2 is a GAN network in which parameters of discriminator is from the file 'nn' and keep static in iteration. when you execute the test_for_GAN.py, the information of network will write in the file'cc3'. 
