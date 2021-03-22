# InSAR-Simulator
InSAR Data Simulator for generating synthetic InSAR noisy images with ground truth clean phase and cohrerence estimation. 

This Simulator is used for paper **"DeepInSAR: A Deep Learning Framework for SAR Interferometric Phase Restoration and Coherence Estimation"**. Please check out the branch ['DeepInSAR-v1'](https://github.com/Lucklyric/InSAR-Simulator/tree/DeepInSAR-v1) to get the same version which is used in paper's experiments. 


## Generate a set of 2D signals
```bash
python gen_2d_signals.py \
   seed=1234 \
   samples=500 \
   workers=5 \
   height=512 \
   width=512 \
   processing_dir=/mnt/hdd1/3vG_data/sim/sim2d/2021-03-21-500_500/
```
