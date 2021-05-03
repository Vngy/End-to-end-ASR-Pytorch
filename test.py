#!/usr/bin/env python
# coding: utf-8
import os
import socket
import sys
import csv
from pathlib import Path
import json
import time
import yaml
import torch
import argparse
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# For reproducibility, comment these may speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str,
                    help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str,
                    help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str,
                    help='Decode output path.', required=False)
parser.add_argument('--load', default=None, type=str,
                    help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed for reproducable results.', required=False)
parser.add_argument('--cudnn-ctc', action='store_true',
                    help='Switches CTC backend from torch to cudnn')
parser.add_argument('--njobs', default=6, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--lm', action='store_true',
                    help='Option for training RNNLM.')
# Following features in development.
parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
parser.add_argument('--reserve-gpu', default=0, type=float,
                    help='Option to reserve GPU ram for training.')
parser.add_argument('--jit', action='store_true',
                    help='Option for enabling jit in pytorch. (feature in development)')
###
paras = parser.parse_args()
setattr(paras, 'gpu', not paras.cpu)
setattr(paras, 'pin_memory', not paras.no_pin)
setattr(paras, 'verbose', not paras.no_msg)
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)

# Hack to preserve GPU ram just incase OOM later on server
if paras.gpu and paras.reserve_gpu > 0:
    buff = torch.randn(int(paras.reserve_gpu*1e9//4)).cuda()
    del buff

#Initialize ASR Solver
mode = 'test'
#Start ASR Server

if paras.test:
    # Test ASR
    assert paras.load is None, 'Load option is mutually exclusive to --test'
    from bin.server_asr import Solver
    mode = 'test'




if(mode == 'test'):
    #Starting Server up
    #model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(constant.args.model_file)
    #start_iter = 0
    #if loaded_args.parallel:
    #    print("unwrap data parallel")
    #    model = model.module
    #audio_conf = dict(sample_rate=loaded_args.sample_rate,
    #              window_size=loaded_args.window_size,
    #              window_stride=loaded_args.window_stride,
    #              window=loaded_args.window,
    #              noise_dir=loaded_args.noise_dir,
    #              noise_prob=loaded_args.noise_prob,
    #              noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

    eprint('Initializing ASR Server')
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('192.168.50.232', 8080)
    eprint('starting up on port {}'.format(server_address))

    sock.bind(server_address)

    sock.listen(5)

    while(True):
        #wait for client
        eprint('Waiting for connection')
        connection, client_address = sock.accept()

        try:
            #Test interaction to client
            msg = 'received'
            sz = connection.recv(1024)
            #Send receive message to client
            connection.sendto(msg.encode(), (client_address))
            #Receive file size from client
            file_size = int(sz.decode("utf-8"))
            encoded_fn = connection.recv(1024)
            connection.sendto(msg.encode(), (client_address))
            #Receive file name from client
            file_name = encoded_fn.decode("utf-8")
            wav_path = './incoming_audio/client/'
            txt_path = './transcriptions/'
            wav_name = file_name.split('.')[-1]+'.wav'
            txt_name = wav_name[:-3]+'txt'
            #clnt = client_thread(connection)
            #clnt.run()
            eprint('connection from:{}'.format(client_address))
            if(os.path.isfile(wav_path+wav_name)):
                os.remove(wav_path+wav_name)
            outfile = open(wav_path + wav_name, 'wb')
          
                
            #receive data in chunks && retransmit
            total_data = 0
            while total_data < file_size:
                data = connection.recv(1024)
                outfile.write(data)
                total_data += len(data)
            outfile.close()
            #write filename to .csv
            Path(txt_path+txt_name).touch()
            eprint('No more Data from: {}\n...\nReceived audio file.\nSending Receipt to Client.'.format(client_address))



            solver = Solver(config, paras, mode)
            solver.load_data()
            solver.set_model()

            #evaluate audio
            results = solver.eval()
            #print out
            print(results)
            #Write to text file on server side
            if(os.path.isfile(txt_path+txt_name)):
                transcribe_file = open(txt_path + txt_name, 'w')
                transcribe_file.write(results[0])
                transcribe_file.close()
            #send trascription to client
            connection.sendto(results[0].encode(), (client_address))
            if (os.path.isfile(wav_path+wav_name)):
                os.remove(wav_path+wav_name)
        finally:
            connection.close()



