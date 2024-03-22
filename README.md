# Deep PDS Learning for AoI Minimization
Source code for paper [Age-Based Scheduling for Mobile Edge Computing: A Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/10449431), written in python and tensorflow.

## Usage
```bash
python DPDS.py
python DPDS.py --Alg='dpl'
python DPDS.py --Alg='coo'
python DPDS.py --Alg='lpo'
```
The running data are (i) recorded by the tf.summary module and can be viewed in real time by running tensorboard in the `logs` directory and (ii) written into matlab format files (`.mat`) in the `data` directory after the simulation is finished.

## Citation
If you find our code helpful, please consider citing our paper.
```
@article{he2024age,
  title={Age-Based Scheduling for Mobile Edge Computing: A Deep Reinforcement Learning Approach},
  author={He, Xingqiu and You, Chaoqun and Quek, Tony QS},
  journal={IEEE Transactions on Mobile Computing},
  year={2024},
  publisher={IEEE}
}
```
