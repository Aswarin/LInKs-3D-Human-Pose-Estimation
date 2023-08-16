# LInKs-3D-Human-Pose-Estimation
This repository contains the code and weights of the lifting models used within the LInKs “Lifting Independent Keypoints” - Partial Pose Lifting for Occlusion
Handling with Improved Accuracy in 2D-3D Human Pose Estimation paper.

We also include the PDF of the supplementary material showing the results of our OpenPose analysis on the Human3.6M dataset.

First, we want to share our appreciation for Bastian Wandt and his help in answering questions during this research.
This work was heavily inspired by Elepose and our code was based upon his obtainable at https://github.com/bastianwandt/ElePose we recommend checking it out.

To train the model or evaluate our weights on the Human3.6M dataset it requires first downloading the dataset with H36m-fetch (https://github.com/anibali/h36m-fetch).
Once you have downloaded the dataset, extracted and preprocessed it you can use our pre_process_h36m_fetch.py file to make it useable with our data loader.

Below is the order you should use the files in:

- 1.) data_utils/pre_process_h36m_fetch.py <- this will preprocess the human3.6m data
- 2.) train_full_pose_norm_flow.py <- This is used to create the full pose norm flow used for generative sampling
- 3.) train_leg_torso_left_right_norm_flow <- This will train the norm flows for each of the independent lifting networks
- 4.) train_left_right_lifter.py + train_leg_torso_lifter.py <- this will train the lifting networks
- 5.) train_occlusion_models.py <- This will train the occlusion models


Note that the code and folder mentioned below still need tidying but send me a message if you need any help and I don't get around to tidying them in time.

I have included some additional files such as viewing the 2D poses drawn in the latent space in data_utils/latent_2d_visualiser

The visualisation folder contains code to visualise the predictions of the lifting networks and align them with the GT to create the videos shown in the videos folder.
The videos folder shows more qualitative results.


