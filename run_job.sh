#!/bin/sh
#export PYTHONPATH:${PYTHONPATH}:/home/dwa6259/grad_work/merkel/newnewLAS/fork/End-to-end-ASR-Pytorch/.
module load gcc-7.4.0-gcc-4.8.5-dan4vbm
module load sox
python3 main.py --config config/libri/echo_asr_ctc.yaml --ckpdir ckpt/elas_libri --logdir log/elas_libri

