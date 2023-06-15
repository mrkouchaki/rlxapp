

import os
import sys
sys.path.append('.')
import schedule
import datetime
import apscheduler
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from zipfile import ZipFile
import json
from os import getenv
from ricxappframe.xapp_frame import RMRXapp, rmr, Xapp
#from mr import sdl

import logging
import numpy as np
import tensorflow as tf
from numpy import zeros, newaxis

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
#from tensorflow.keras import losses
import gym

import numpy as np
import pandas as pd
import statistics
from statistics import mean
import matplotlib.pyplot as plt
import IPython
from IPython import display

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense  
from tensorflow.keras.layers import Activation  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from _thread import *
import socket
import threading
import errno
import struct
import pickle


xapp = None
pos = 0
RAN_data = None
rmr_xapp = None

DISCONNECT_MESSAGE = "!DISCONNECT"

reg_actor_list = []

scheduler = BackgroundScheduler()

class UENotFound(BaseException):
    pass
class CellNotFound(BaseException):
    pass

def post_init(self):
    print('///////enter def post_init__/////////////////')
    """
    Function that runs when xapp initialization is complete
    """
    self.def_hand_called = 0
    self.traffic_steering_requests = 0


def handle_config_change(self, config):
    print('////////enter def handle_config_change//////////////')
    """
    Function that runs at start and on every configuration file change.
    """
    self.logger.debug("handle_config_change: config: {}".format(config))


def default_handler(self, summary, sbuf):
    print('/////////enter def default_handler///////////////')
    """
    Function that processes messages for which no handler is defined
    """
    self.def_hand_called += 1
    print('self.def_hand_called += 1=', self.def_hand_called)
    self.logger.warning("default_handler unexpected message type {}".format(summary[rmr.RMR_MS_MSG_TYPE]))
    self.rmr_free(sbuf)

 
class CentralCritic(models.Model):
    def __init__(self, state_shape, action_shape):
        super(CentralCritic, self).__init__()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(1, activation=None)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)
    
    def print_input_output_dim(self):
        print("Input shape:", self.input_shape)
        print("Output shape:", self.output_shape)
    
    def print_input_output_shape(self):
        print("Input shape:", self.input.shape)
        print("Output shape:", self.output.shape)
    
    def print_summary(self):
        self.summary()
        
class AdvancedFederatedRL_critic:
    def __init__(self, N, M, L, state_shape, action_shape, listen_port=12345, buffer_size=100000, batch_size=10, gamma=0.99, tau=0.005, learn_iterations=1,
                 alpha=0.6, lr_start=0.0001, lr_decay_steps=100000, lr_end=0.00001, 
                 beta_start=0.4, beta_end=1.0, beta_anneal_steps=200000):
        self.N = N
        self.L = L
        self.M = M
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learn_iterations = learn_iterations
        
        self.lr_start = lr_start
        self.lr_decay_steps = lr_decay_steps
        self.lr_end = lr_end

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps

        self.port = listen_port
        self.listen_port = listen_port
        
        self.central_critic = CentralCritic(state_shape, action_shape)
        self.target_central_critic = CentralCritic(state_shape, action_shape)
        self.update_target_critic(tau=1.0)

        self.learning_rate_schedule = optimizers.schedules.PolynomialDecay(self.lr_start, self.lr_decay_steps, self.lr_end)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate_schedule)
                           

    def compute_target_Q_values(self, data):
        print('im in compute_target_Q_values')
        states, actions, next_states, rewards, dones, next_actions = data["states"], data["actions"], data["next_states"], data["rewards"], data["dones"], data["next_actions"]
        
        target_q_values = self.central_critic(next_states, next_actions)
        done_mask = tf.cast(tf.reshape(1 - dones, [-1, 1]), tf.float32)
        rewards = tf.reduce_sum(rewards, axis=1, keepdims=True)
        rewards = tf.cast(rewards, tf.float32)
        target_q_values = tf.stop_gradient(rewards + self.gamma * done_mask* target_q_values)
        print('target_q_values:',target_q_values)
        
        #target_Q_values = []
        #for i, target_critic in enumerate(self.target_central_critics):
            #next_Q_values = target_critic(next_states, next_actions)
            #target_Q_value = rewards[:, i] + (1 - dones) * self.gamma * next_Q_values
            #target_Q_values.append(target_Q_value)
        #target_Q_values = np.stack(target_Q_values, axis=-1)

        return target_q_values

    
    def update_target_critic(self, tau=None):
        print('im in update_target_critic')
        if tau is None:
            tau = self.tau
        target_critic_weights = self.target_central_critic.get_weights()
        critic_weights = self.central_critic.get_weights()
        new_weights = [tau * w + (1.0 - tau) * tw for w, tw in zip(critic_weights, target_critic_weights)]
        self.target_central_critic.set_weights(new_weights)

    def compute_critic_gradients(self, data):
        print('im in compute_critic_gradients')
        states, actions, weights, target_Q_values = data["states"], data["actions"], data["weights"], data["target_q_values"]
        
        with tf.GradientTape() as tape:
            q_values = self.central_critic(states, actions)
            #print('q_values = self.central_critic(states, actions):', q_values)
            td_errors = target_Q_values - q_values
            #print('weights * tf.square(td_errors):', weights * tf.square(td_errors))
            #********hatman check, weights:horizontal(1,10), td-error:vertical(10,1)               
            critic_loss = tf.reduce_mean(weights * tf.square(td_errors))
            #print('critic_loss:', critic_loss)
        critic_gradients = tape.gradient(critic_loss, self.central_critic.trainable_variables)
        #print('critic_gradients:', critic_gradients)
        #self.optimizer.apply_gradients(zip(critic_gradients, self.central_critic.trainable_variables))

        return critic_gradients
    
    def compute_Q_values(self, data):
        print('im in compute_Q_values')
        states, actions = data["states"], data["actions"]
        Q_values = []
        with tf.GradientTape() as tape:
            q_values_i = self.central_critic(states, actions)
            print('q_values_i:', q_values_i)
        return q_values_i
            
    
    def learn(self, data):
        print('im in learn')
        states, actions, target_Q_values = data["states"], data["actions"], data["target_q_values"]

        critic_gradients = self.compute_critic_gradients(data)

        # Apply gradients to the critic networks
        self.optimizer.apply_gradients(zip(critic_gradients, self.central_critic.trainable_variables))
        #for i, critic in enumerate(self.critics):
            #critic.optimizer.apply_gradients(zip(critic_gradients[i], critic.trainable_variables))

        # Update target critic networks
        self.update_target_critic()
    
    def get_weights(self):
        print('im in get_weights')
        critic_weights = self.central_critic.get_weights()
        return critic_weights

    def set_weights(self, critic_weights):
        print('im in set_weights')
        self.central_critic.set_weights(critic_weights)
    
    def update_target_critic(self, tau=None):
        print('im in update_target_critic')
        if tau is None:
            tau = self.tau
        target_critic_weights = self.target_central_critic.get_weights()
        critic_weights = self.central_critic.get_weights()
        new_weights = [tau * w + (1.0 - tau) * tw for w, tw in zip(critic_weights, target_critic_weights)]
        self.target_central_critic.set_weights(new_weights)
      
def get_machine_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def start_critic_server(critic_instance, ip="0.0.0.0", port=8585):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('port in start_critic_server:', port)
    server.bind((ip, port))
    server.listen()

    # Get the actual port number assigned by the OS
    actual_port = server.getsockname()[1]

    machine_ip = get_machine_ip()
    print(f"Machine IP address: {machine_ip}")
    print(f"Critic Server listening on {ip}:{actual_port}")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")

        client_thread = threading.Thread(target=handle_client, args=(client_socket, critic_instance, machine_ip, actual_port))
        client_thread.start()

def handle_client(client_socket, critic_instance, machine_ip, actual_port):
    print('im in handle_client')
    try:
        while True:
            received_data = b""
            while len(received_data) < 4:
                more_data = client_socket.recv(4 - len(received_data))
                if not more_data:
                    break
                received_data += more_data
            if not received_data:
                break

            data_size = struct.unpack(">I", received_data)[0]
            print(f"Server-side received data_size: {data_size}")

            chunk_size = 65536
            received_data = b""
            while len(received_data) < data_size:
                more_data = client_socket.recv(min(data_size - len(received_data), chunk_size))
                if not more_data:
                    raise IOError("Connection closed while receiving data")
                received_data += more_data

            request = pickle.loads(received_data)
            operation = request["operation"]
            print(f"Operation: {operation}")

            if operation == "get_target_Q_values":
                target_Q_values = critic_instance.compute_target_Q_values(request["data"])
                response = target_Q_values

            elif operation == "update_critic":
                critic_instance.learn(request["data"])
                response = "Critic Updated"

            elif operation == "get_Q_values":
                print('operation received: get_Q_values')
                Q_values = critic_instance.compute_Q_values(request["data"])
                response = Q_values

            elif operation == "get_critic_weights":
                print('operation received: get_critic_weights')
                critic_weights = critic_instance.get_weights()
                response = critic_weights

            response_data = pickle.dumps(response)
            response_data_size = len(response_data)
            client_socket.sendall(struct.pack(">I", response_data_size))
            client_socket.sendall(response_data)
            print('----------------------------------------------------/')
            print(f"Critic Server listening on {machine_ip}:{actual_port}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        client_socket.close()
        
           
#     ue_data = pd.DataFrame(ue_data_kpimon)
#     cell_data = pd.DataFrame(cell_data_kpimon)
#     print('ue_data=', ue_data)
#     print('cell_data=', cell_data)
#     ue_data = time(ue_data)
#     cell_data = time(cell_data)
#     print('df2 = time(ue_data)=', ue_data)
#     print('df3 = time(cell_data)=', cell_data)
    
    
# def waiting_actor_thread(actor_socket):
#     while True:
#         data = actor_socket.recv(4096)
#         data = data.decode("utf-8")
#         print("<<-- Received message: {}".format(data) + " from the actor")
#         #if not data:
#         if len(data) == 0:
#             print("Actor disconnected!")
#             break


def start(thread=False):
 
    print('////////////////entered Starrrrrrrrrrrt///////////////////')
    """
    This is a entry point function in dockerized xapp
    """
    global xapp

    #xapp = Xapp(entrypoint=entry, rmr_port=4560, use_fake_sdl=False)
    #print('xapp = Xapp(entrypoint=entry, rmr_port=4560, use_fake_sdl=fake_sdl)=', xapp)
  
    use_fake_sdl=False
    rmr_port=4560
    
    #start_server_listening()
    #print("[STARTING] server is start listening...")
    
    state_shape = 28  # Should Replace with environment state_shape
    action_shape = 14  # Should Replace with environment action_shape
    N = 2
    M = 4
    L = 7
    critic_instance = AdvancedFederatedRL_critic(N, M, L, state_shape, action_shape)
    machine_ip = get_machine_ip()
    print(f"Machine IP address: {machine_ip}")
    start_critic_server(critic_instance, machine_ip)
    


def stop():
    print('/////////////enter def stop//////////////////')      
    """
    can only be called if thread=True when started
    """
    xapp.stop()


def get_stats():
    print('//////////////////enter def get_stats()////////////////////')
    """
    hacky for now, will evolve
    """
    print('DefCalled:rmr_xapp.def_hand_called=', rmr_xapp.def_hand_called)
    print('SteeringRequests:rmr_xapp.traffic_steering_requests=', rmr_xapp.traffic_steering_requests) 
    return {"DefCalled": rmr_xapp.def_hand_called,
            "SteeringRequests": rmr_xapp.traffic_steering_requests}
 


def mr_req_handler(self, summary, sbuf):
    print('///////////enter def mr_req handler/////////////')
    """
    This is the main handler for this xapp, which handles load prediction requests.
    This app fetches a set of data from SDL, and calls the predict method to perform
    prediction based on the data

    The incoming message that this function handles looks like:
        {"UEPredictionSet" : ["UEId1","UEId2","UEId3"]}
    """
    #self.traffic_steering_requests += 1
    # we don't use rts here; free the buffer
    self.rmr_free(sbuf)

    ue_list = []
    try:
        print('////enter first try in mr_req_handler////')
        print('rmr.RMR_MS_PAYLOAD=', rmr.RMR_MS_PAYLOAD)
        print('summary[rmr.RMR_MS_PAYLOAD]=', summary[rmr.RMR_MS_PAYLOAD])
        req = json.loads(summary[rmr.RMR_MS_PAYLOAD])  # input should be a json encoded as bytes
        print('req = json.loads(summary[rmr.RMR_MS_PAYLOAD])=', req)
        ue_list = req["UEPredictionSet"]
        print('ue_list=req["UEPredictionSet"] =', ue_list)
        self.logger.debug("mr_req_handler processing request for UE list {}".format(ue_list))
    except (json.decoder.JSONDecodeError, KeyError):
        print('////enter first except in mr_req_handler////')
        self.logger.warning("mr_req_handler failed to parse request: {}".format(summary[rmr.RMR_MS_PAYLOAD]))
        return
    print('ue_list mr_req_handler aftr 1st try=', ue_list)
    # iterate over the UEs, fetches data for each UE and perform prediction
    for ueid in ue_list:
        try:
            print('////enter second try in mr_req_handler////')
            uedata = sdl.get_uedata(self, ueid)
            print('uedata = sdl.get_uedata(self, ueid)=', uedata)
            predict(self, uedata)
            print('predict(self, uedata)=', predict(self, uedata))
        except UENotFound:
            print('////enter second except in mr_req_handler////')
            print('enter UENotFound in mr_req_handler')
            self.logger.warning("mr_req_handler received a TS Request for a UE that does not exist!")    
    
 

