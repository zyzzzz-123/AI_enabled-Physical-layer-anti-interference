# AI_enabled-Physical-layer-anti-interference
This is a course project in **Principles of Wireless 
Communication and Mobile Networks** (IE304 in SJTU). 
We aim to utilize autoencoder to detect the occurrence 
of disturbances or collisions and avoid them, 
and further realize whole anti-interference communication technology.

## Group Members
**YUZHE ZHANG** (group leader) 

**SHANMU WANG, YOUSONG ZHANG, ZEXI LIU**

**TA: MINGQI XIE**
# Still In Process

## Update 2022.5.17 One type One channel Situaion
1. One channel: only 1 of 64 in *encoded* (whose physical meaning 
is 64 channels of different frequency)
2. One type of interference: There is only 1 pattern of the structural interference, which is 
generated in Matlab, which means that the output size of observation is also 1.
3. Overwrite the model and rectify some bugs. The constellation figure of encoded and interference in 
frequency domain is shown as below.

![Figure 2](Figure_one_channel_one_interference.png)

## Update 2022.4.26 Restruct Framework
Reformation of a trivial manipulative frame, which include trainer and channel.
We load channel data from .mat and add channel interference on them.

And we are still working on adding the AF module.
## Update 2022.4.10 Next Stage Target
After the base module of autoencoder and various channels has been implemented, 
we need to add appropriate channel detection and feedback modules to our work.

After study of related work, we find a paper ***Wireless Image Transmission Using Deep Source
Channel Coding With Attention Modules***, which propose to exploit some AF modules.
We will endeavor to improve the existing work under the guidance of following illustration.

![Figure 1](images/1.jpg)

## Update 2022.3.30 Some experiment results ##
Here is some display of recent experiment results.

#### 1.Figure1. Accurary Under AWGN of Different SNR
![test image size](images/2.jpg)

#### 2.Figure2. Accurary Under Strutucal Damage
![](images/3.jpg)

#### 3.Figure3. Accurary Under CFO
![](images/4.jpg)

#### 4.Figure4. Accurary Under SFO
![](images/5.jpg) 

## Update 2022.3.1 Division of Tasks and Initial Realization##
The target schedule and work distribution of out project in the first stage is shown as followings.
![](images/6.jpg)

## Update 2022.2 Work distribution 
![](images/7.jpg)