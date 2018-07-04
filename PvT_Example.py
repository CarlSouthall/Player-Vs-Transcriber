"""
@author: Carl Southall     carlsouthall.com    carl.southall@bcu.ac.uk     https://github.com/CarlSouthall
"""

# This file contains an example implementation of the AugAddExist player vs transcriber model from [1].

# The player model uses a single two layered CNN and the transcriber model consists of the cnnSA3F5 network from [1,2] which consists of two conv layers, two LSTMP BRNN layers and then a
# soft attention mechanism output layer. The WMD multiple time-step loss function from [2] is used.

# This example uses a single sample addition loop for simplicity. The various stages of the player model 
# are plotted in Figure 1 and Figure 2 presents the output activation function against the ground truth.
#

#  [1] Southall, Carl, Ryan Stables and Jason Hockman. 2018. 
#  Player Vs Transcriber: A Game Approach to Data Manipulation for Automatic Drum Transcription.
#  In Proceedings of the 19th International Society for Music Information
#  Retrieval Conference (ISMIR), Paris, France, 2018.
#
#  [2] Southall, Carl, Ryan Stables and Jason Hockman. 2018. 
#  Improving Peak-picking Using Multiple Time-step Loss Functions. 
#  In Proceedings of the 19th International Society for Music Information
#  Retrieval Conference (ISMIR), Paris, France, 2018.

#################################################################################
# packages and variables

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.contrib import rnn 
import matplotlib.pyplot as plt  
import os

eps=np.finfo(float).eps

#################################################################################

## loss functions proposed in [2]

def cross_entropy_form(pred,lab):
    out=(lab * tf.log(pred))+((1-lab)*(tf.log(1-pred)))
    return out

def mean_squared_form(pred,lab):
    out=tf.square(lab-pred)
    return out

def mean_squared_peak_dif_form(pred1,lab1,pred2,lab2):
    out=tf.square((lab1-lab2)-(pred1-pred2))
    return out

def weighted_cross_entropy_peak_dif_form(pred1,lab1,pred2,lab2,FP_weighting,eps=np.finfo(float).eps):
    FN_weighting=1-FP_weighting
    out=FN_weighting*(tf.abs(lab1-lab2)*(tf.log(tf.abs(pred1-pred2)+eps)))+FP_weighting*((1-tf.abs(lab1-lab2))*(tf.log(1-tf.abs(pred1-pred2)+eps)))
    return out 

def MI(preds,labs,weighting,seq_len):
   stand_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,1:seq_len+1],labs[:,1:seq_len+1]),dtype=(tf.float32))    
   previous_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,:seq_len],labs[:,:seq_len]),dtype=(tf.float32))
   future_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]), (preds[:,2:seq_len+2],labs[:,2:seq_len+2]),dtype=(tf.float32))
   cost=tf.reduce_mean(tf.add(tf.reshape(stand_ce,[-1]),(weighting*tf.add(tf.reshape(future_ce,[-1]),tf.reshape(previous_ce,[-1])))))

   return cost

def MMD(preds,labs,weighting,seq_len):      
   stand_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,1:seq_len+1],labs[:,1:seq_len+1]),dtype=(tf.float32))    
   previous_ce=tf.map_fn(lambda x: -tf.reduce_sum(mean_squared_peak_dif_form(x[0],x[1],x[2],x[3]), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,:seq_len],labs[:,:seq_len]),dtype=(tf.float32))
   future_ce=tf.map_fn(lambda x: -tf.reduce_sum(mean_squared_peak_dif_form(x[0],x[1],x[2],x[3]), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,2:seq_len+2],labs[:,2:seq_len+2]),dtype=(tf.float32))
   cost=tf.reduce_mean(tf.add(tf.reshape(stand_ce,[-1]),(weighting*tf.add(tf.reshape(future_ce,[-1]),tf.reshape(previous_ce,[-1])))))
   return cost

def WMD(preds,labs,weighting,seq_len,FP_weighting_):      
   stand_ce=tf.map_fn(lambda x: -tf.reduce_sum(cross_entropy_form(x[0],x[1]), reduction_indices=[1]),(preds[:,1:seq_len+1],labs[:,1:seq_len+1]),dtype=(tf.float32))    
   previous_ce=tf.map_fn(lambda x: -tf.reduce_sum(weighted_cross_entropy_peak_dif_form(x[0],x[1],x[2],x[3],FP_weighting_), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,:seq_len],labs[:,:seq_len]),dtype=(tf.float32))
   future_ce=tf.map_fn(lambda x: -tf.reduce_sum(weighted_cross_entropy_peak_dif_form(x[0],x[1],x[2],x[3],FP_weighting_), reduction_indices=[1]), (preds[:,1:seq_len+1],labs[:,1:seq_len+1],preds[:,2:seq_len+2],labs[:,2:seq_len+2]),dtype=(tf.float32))
   cost=tf.reduce_mean(tf.add(tf.reshape(stand_ce,[-1]),(weighting*tf.add(tf.reshape(future_ce,[-1]),tf.reshape(previous_ce,[-1])))))
   return cost


#############################

## PvT Model Used in [1]

class CNNSA3F5_AugAddExist():
       
     def __init__(self,training_data=[], training_labels=[], validation_data=[], validation_labels=[], network_save_filename=[], minimum_epoch=5, maximum_epoch=10, n_hidden=[100,100], n_classes=2, attention_number=2, dropout=0.75,  learning_rate=0.003 ,save_location=[],snippet_length=100, cost_type='CE',batch_size=1000,input_feature_size=84,conv_filter_shapes=[[3,3,1,32],[3,3,32,64]], conv_strides=[[1,1,1,1],[1,1,1,1]], pool_window_sizes=[[1,3,3,1],[1,3,3,1]],
                  sample_spec_path='ExampleSamples.npy',sample_aug_val=0,sample_num_locations=75,no_sample_ins=3,sample_num_batch=93,eps=np.finfo(np.float32).eps,aug_amp_min=0.5,peak_distance=3,player_conv_filter_shapes=[[3,3,1,5],[3,3,5,10]],player_pool_window_sizes=[[1,8,7,1],[1,7,7,1]],player_conv_strides=[[1,1,1,1],[1,1,1,1]],player_on='yes'):         

         self.num_samples_per_snippet=1
         self.features=training_data
         self.targ=training_labels
         self.val=validation_data
         self.val_targ=validation_labels
         self.filename=network_save_filename
         self.n_hidden=n_hidden
         self.n_layers=len(self.n_hidden)
         self.dropout=dropout
         self.truncated=snippet_length
         self.learning_rate=learning_rate
         self.n_classes=n_classes
         self.minimum_epoch=minimum_epoch
         self.maximum_epoch=maximum_epoch
         self.num_batch=int(len(self.features)/batch_size)
         self.val_num_batch=int(len(self.val)/batch_size)
         self.batch_size=batch_size
         self.attention_number=attention_number
         self.cost_type=cost_type
         self.input_feature_size=input_feature_size
         self.batch=np.zeros((self.batch_size,self.input_feature_size))
         self.batch_targ=np.zeros((self.batch_size,self.n_classes))
         self.save_location=save_location
         self.snippet_length=snippet_length
         self.num_seqs=int(self.batch_size/self.snippet_length)
         self.conv_filter_shapes=conv_filter_shapes
         self.conv_strides=conv_strides
         self.pool_window_sizes=pool_window_sizes
         self.pool_strides=self.pool_window_sizes
         self.pad='SAME'
         self.sample_specs=np.load(sample_spec_path)
         self.sample_aug_value=sample_aug_val
         self.sample_num_locations=sample_num_locations
         self.no_sample_ins=no_sample_ins
         self.sample_len=int(self.sample_specs[0][0].shape[0])
         self.num_samples_per_batch=self.num_seqs
         self.sample_num_batch=sample_num_batch
         self.eps=eps
         self.player_n_hidden=(((2*self.input_feature_size)+1)*(self.num_samples_per_snippet+1))+((self.sample_num_locations*self.no_sample_ins)*self.num_samples_per_snippet)
         self.player_conv_filter_shapes=player_conv_filter_shapes
         self.player_conv_strides=player_conv_strides
         self.player_pool_window_sizes=player_pool_window_sizes
         self.aug_amp_min=aug_amp_min    
         self.peak_distance=peak_distance
         self.player_on=player_on 
         
         self.player_conv_layer_out=[]
         self.player_h_conv=[]
         self.player_w_conv=[]
         self.player_b_conv=[]
         self.conv_layer_out=[]
         self.fc_layer_out=[]
         self.w_fc=[]
         self.b_fc=[]
         self.h_fc=[]
         self.w_conv=[]
         self.b_conv=[]
         self.h_conv=[]
         self.h_pool=[]
         self.h_drop_batch=[]
         
         
     def cell_create(self,scope_name):
         with tf.variable_scope(scope_name):
             if int(scope_name)==1:
                 cells = rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[int(scope_name)-1]) for i in range(1)], state_is_tuple=True)
                 cells = rnn.DropoutWrapper(cells, input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph)
             else:
                 cells = rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[int(scope_name)-1+i]) for i in range(self.n_layers-1)], state_is_tuple=True)
                 cells = rnn.DropoutWrapper(cells, input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph)                 
         return cells
     
     def weight_bias_init(self):
               
            self.biases = tf.Variable(tf.zeros([self.n_classes]))           
            self.weights =tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2, self.n_classes]))

     def cell_create_norm(self):
         cells = rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
         cells = rnn.DropoutWrapper(cells, input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph) 
         return cells
           
     def attention_weight_init(self,num):
         if num==0:
             self.attention_weights=[tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*4,self.n_hidden[(len(self.n_hidden)-1)]*2]))]
             self.sm_attention_weights=[tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2,self.n_hidden[(len(self.n_hidden)-1)]*2]))]
         if num>0:
             self.attention_weights.append(tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*4,self.n_hidden[(len(self.n_hidden)-1)]*2])))
             self.sm_attention_weights.append(tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*2,self.n_hidden[(len(self.n_hidden)-1)]*2])))

                                  
     def conv2d(self,data, weights, conv_strides, pad):
         return tf.nn.conv2d(data, weights, strides=conv_strides, padding=pad)
     
     def max_pool(self,data, max_pool_window, max_strides, pad):
        return tf.nn.max_pool(data, ksize=max_pool_window,
                            strides=max_strides, padding=pad)
        
     def weight_init(self,weight_shape):
        weight=tf.Variable(tf.truncated_normal(weight_shape))    
        return weight
        
     def bias_init(self,bias_shape,):   
        bias=tf.Variable(tf.constant(0.1, shape=bias_shape))
        return bias
    
     def batch_dropout(self,data):
        batch_mean, batch_var=tf.nn.moments(data,[0])
        scale=tf.Variable(tf.ones([self.num_seqs*self.seq_len,data.get_shape()[1],data.get_shape()[2],data.get_shape()[3]]))
        beta=tf.Variable(tf.zeros([self.num_seqs*self.seq_len,data.get_shape()[1],data.get_shape()[2],data.get_shape()[3]]))
        h_poolb=tf.nn.batch_normalization(data,batch_mean,batch_var,beta,scale,1e-3)
        return tf.nn.dropout(h_poolb, self.dropout_ph)
    
     def weight_init_player(self,weight_shape,name):
        weight=tf.Variable(tf.truncated_normal(weight_shape),name=name+'_w')    
        return weight
        
     def bias_init_player(self,bias_shape,name):   
        bias=tf.Variable(tf.constant(0.1, shape=bias_shape),name=name+'_b')
        return bias
        
     def conv_2dlayer(self,layer_num):
        self.w_conv.append(self.weight_init(self.conv_filter_shapes[layer_num]))
        self.b_conv.append(self.bias_init([self.conv_filter_shapes[layer_num][3]]))
        self.h_conv.append(tf.nn.relu(self.conv2d(self.conv_layer_out[layer_num], self.w_conv[layer_num], self.conv_strides[layer_num], self.pad) + self.b_conv[layer_num]))
        self.h_pool.append(self.max_pool(self.h_conv[layer_num],self.pool_window_sizes[layer_num],self.pool_strides[layer_num],self.pad))       
#        self.conv_layer_out.append(self.batch_dropout(self.h_pool[layer_num]))  # removed for quicker run time
        self.conv_layer_out.append(self.h_pool[layer_num])  

     def reshape_layer(self):
            convout=self.conv_layer_out[len(self.conv_layer_out)-1]
            self.fc_layer_out=tf.reshape(convout, [self.num_seqs,self.seq_len,self.conv_filter_shapes[1][3]*(np.ceil(np.ceil(self.input_feature_size/float(self.pool_window_sizes[0][2]))/float(self.pool_window_sizes[1][2])))*2])
              
     def player_conv_2dlayer(self,layer_num):
        self.player_w_conv.append(self.weight_init_player(self.player_conv_filter_shapes[layer_num],'player_conv'+str(layer_num)))
        self.player_b_conv.append(self.bias_init_player([self.player_conv_filter_shapes[layer_num][3]],'player_conv'+str(layer_num)))
        
        if layer_num==0:
            self.player_conv_layer_out.append(tf.reshape(tf.concat((tf.squeeze(self.snippets1),tf.reshape(self.snippets_samples1,[self.num_samples_per_batch,-1,self.input_feature_size])),1),[self.num_samples_per_batch,-1,self.input_feature_size,1]))
        self.player_h_conv.append(tf.nn.relu(self.conv2d(self.player_conv_layer_out[layer_num], self.player_w_conv[layer_num], self.player_conv_strides[layer_num],'SAME') + self.player_b_conv[layer_num]))
        self.player_conv_layer_out.append(self.max_pool(self.player_h_conv[layer_num],self.player_pool_window_sizes[layer_num],self.player_pool_window_sizes[layer_num],'SAME'))  

     def player_fc_layer(self):
        self.player_fc_layer_in=tf.reshape(self.player_conv_layer_out[len(self.player_conv_layer_out)-1], [self.num_samples_per_batch, int(np.ceil(np.ceil(self.input_feature_size/float(self.player_pool_window_sizes[0][2]))/float(self.player_pool_window_sizes[1][2]))*self.player_conv_filter_shapes[1][3]*np.ceil(np.ceil((self.snippet_length+(self.no_sample_ins*self.sample_len))/float(self.player_pool_window_sizes[0][1]))/float(self.player_pool_window_sizes[1][1])))])
        self.player_h_fc=tf.nn.relu(tf.matmul(self.player_fc_layer_in, self.player_w_fc) + self.player_b_fc)
        self.player_fc_layer_out=tf.nn.dropout(self.player_h_fc, self.dropout_ph)
            
     # create PvT Model
     def create(self):
       
       tf.reset_default_graph()
       
       self.sample_tv_switch=tf.placeholder("int32")
       self.targets = tf.placeholder("float32", [None, None, self.batch_targ.shape[1]]) 
       self.snippets_existing = tf.placeholder("float32", [None, None, None, self.input_feature_size]) 
       self.snippets_samples = tf.placeholder("float32", [None, None, None, self.input_feature_size])     

       #create player model output layer weights
       
       self.player_w_fc=self.weight_init_player([int(np.ceil(np.ceil(self.input_feature_size/float(self.player_pool_window_sizes[0][2]))/float(self.player_pool_window_sizes[1][2]))*self.player_conv_filter_shapes[1][3]*np.ceil(np.ceil((self.snippet_length+(self.no_sample_ins*self.sample_len))/float(self.player_pool_window_sizes[0][1]))/float(self.player_pool_window_sizes[1][1]))), self.player_n_hidden],'player_fc') 
       self.player_b_fc=self.bias_init_player([self.player_n_hidden],'player_fc')
       self.dropout_ph=0.75

       self.snippets1 = tf.cond(self.sample_tv_switch < 1, lambda: self.snippets_existing, lambda: tf.zeros((1,self.num_samples_per_batch,self.snippet_length,self.input_feature_size)))
       self.snippets_samples1 = tf.cond(self.sample_tv_switch < 1, lambda: self.snippets_samples, lambda: tf.zeros((self.num_samples_per_batch,self.no_sample_ins, self.sample_len,self.input_feature_size)))
      
       
       # get the outputs from the player network
       
       for i in range(len(self.player_conv_filter_shapes)):
           self.player_conv_2dlayer(i)
       self.player_fc_layer()
       self.j=tf.range(self.snippet_length)
       [self.loc_fc_layer_out,self.aug_fc_layer_out]=tf.split(self.player_fc_layer_out,[((self.sample_num_locations*self.no_sample_ins)*self.num_samples_per_snippet),(((2*self.input_feature_size)+1)*(self.num_samples_per_snippet+1))],1)
       
       self.sample_aug_vals=(self.input_feature_size*2)+1
       self.loc_fc_layer_out=tf.split(self.loc_fc_layer_out,[self.sample_num_locations*self.no_sample_ins],1)  
       self.aug_fc_layer_out=tf.split(self.aug_fc_layer_out,[self.sample_aug_vals,self.sample_aug_vals],1)
       self.aug_fc_layer_out[0]=tf.split(self.aug_fc_layer_out[0],[1,2*self.input_feature_size],1)
       self.aug_fc_layer_out[1]=tf.split(self.aug_fc_layer_out[1],[1,2*self.input_feature_size],1)
          
       # augment existing data
       self.snippets_p_exist=tf.map_fn(lambda x: (tf.nn.relu(x[1]+((tf.cast(tf.nn.softmax(x[0][0],0),tf.float32)*x[1]*self.sample_aug_value))-(tf.cast(tf.nn.softmax(x[0][1],0),tf.float32)*x[1]*self.sample_aug_value)))*(x[2]*(1-self.aug_amp_min)+self.aug_amp_min),(tf.tile(tf.expand_dims(tf.reshape(self.aug_fc_layer_out[0][1],[self.num_samples_per_batch,2,-1]),2),[1,1,self.snippet_length,1]),self.snippets1[0],tf.nn.sigmoid(self.aug_fc_layer_out[0][0])),dtype=(tf.float32))              
       
       self.locations3 = tf.cond(self.sample_tv_switch < 1, lambda: self.targets, lambda: tf.zeros((self.num_samples_per_batch,self.snippet_length,self.no_sample_ins)))
       self.spectrums=self.snippets_p_exist
       
       # sample addition stage
       #Example is limited to adding one sample for simplicity to add more samples just repeat this stage
       
       # augment samples
       
       self.snippets_p1=tf.reshape(tf.map_fn(lambda x: (tf.nn.relu(x[1]+((tf.cast(tf.nn.softmax(x[0][0],0),tf.float32)*x[1]*self.sample_aug_value))-(tf.cast(tf.nn.softmax(x[0][1],0),tf.float32)*x[1]*self.sample_aug_value)))*(x[2]*(1-self.aug_amp_min)+self.aug_amp_min),(tf.tile(tf.expand_dims(tf.reshape(self.aug_fc_layer_out[1][1],[self.num_samples_per_batch,2,-1]),2),[1,1,self.sample_len*self.no_sample_ins,1]),tf.reshape(self.snippets_samples1,[self.num_samples_per_batch,-1,self.input_feature_size]),tf.nn.sigmoid(self.aug_fc_layer_out[1][0])),dtype=(tf.float32)),[self.num_samples_per_batch,self.no_sample_ins,self.sample_len,self.input_feature_size])  
       self.snippets_padded1=tf.split(tf.transpose(tf.reshape(tf.transpose(tf.map_fn(lambda x:tf.pad(self.snippets_p1,[[0,0],[0,0],[x,self.sample_num_locations-x+8],[0,0]]),tf.range(self.sample_num_locations),dtype=tf.float32),[1,2,0,3,4]),[self.num_samples_per_batch,self.no_sample_ins*self.sample_num_locations,self.sample_len+self.sample_num_locations+8,self.input_feature_size]),[0,2,3,1]),[8,self.sample_len+self.sample_num_locations],1)[1]
       
       # generate sample location
       
       self.locations11=tf.map_fn(lambda x:tf.divide(tf.transpose(x,[1,0]),tf.reduce_max(x,1)),tf.reshape(self.loc_fc_layer_out[0],[self.num_samples_per_batch,self.no_sample_ins,self.sample_num_locations]),dtype=tf.float32)
       self.locations12=tf.map_fn(lambda x: tf.subtract(tf.add(x,self.eps),tf.reduce_max(x,0)),self.locations11,dtype=tf.float32)
       self.locations13=tf.pad(tf.map_fn(lambda x: tf.multiply(tf.nn.relu(x),tf.divide(1,self.eps)),self.locations12,dtype=tf.float32),[[0,0],[0,self.snippet_length-self.sample_num_locations],[0,0]])
       
       self.locations3_padded=tf.transpose(tf.pad(self.locations3,[[0,0],[self.peak_distance+1,self.peak_distance+1],[0,0]]),[1,0,2])
       self.loc_vals1=tf.transpose(tf.reduce_mean(tf.map_fn(lambda x: tf.split(self.locations3_padded,[1+x,(self.peak_distance*2)+1,self.snippet_length-x],0)[1],self.j,dtype=tf.float32),1),[1,0,2])
       self.loc_vals1_reshape=tf.reshape(tf.transpose(self.loc_vals1,[0,2,1]),[self.snippet_length,-1])
       self.loc_vals1_m=tf.multiply(tf.divide(tf.reduce_max(self.loc_vals1_reshape,0),self.loc_vals1_reshape),tf.subtract(self.loc_vals1_reshape,self.eps))
       self.loc_valsf1=tf.transpose(tf.reshape(1-tf.nn.relu((tf.divide(self.loc_vals1_m,tf.reduce_max(self.loc_vals1_m,0)))),[self.num_samples_per_batch,self.no_sample_ins,self.snippet_length]),[0,2,1])
#           
       self.locations14=tf.multiply(self.locations13,self.loc_valsf1)

       
       self.spectrums1=tf.split(tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0],tf.concat((x[1][:,0],x[1][:,1],x[1][:,2]),0)),2),(self.snippets_padded1, tf.split(self.locations14,[self.sample_num_locations,-1],1)[0]),dtype=tf.float32),[self.snippet_length,-1],1)[0]
       
       # add new samples to existing data
       
       self.overall_spectrum=self.spectrums+self.spectrums1
       self.locations_overall=self.locations3+self.locations14
       
       self.locations_zero=self.locations_overall
       self.locations_zero.set_shape([self.num_samples_per_batch,self.snippet_length,self.no_sample_ins])


       self.x_ph = tf.cond(self.sample_tv_switch < 1, lambda: self.overall_spectrum ,lambda: self.snippets_existing[0])
       self.x_ph.set_shape([None,None,self.input_feature_size])

       self.y_ph = tf.cond(self.sample_tv_switch < 1, lambda: self.locations_zero, lambda: self.targets)
       
       # create transcriber model
       
       
       self.seq=tf.placeholder("int32")
       self.num_seqs=tf.placeholder("int32")
       self.seq_len=tf.placeholder("int32")
       self.dropout_ph = tf.placeholder("float32")
       
       self.x_ph=tf.pad(self.x_ph,[[0,0],[6,6],[0,0]])
       
       self.conv_layer_out.append(tf.expand_dims(tf.reshape(tf.transpose(tf.map_fn(lambda x:tf.split(self.x_ph,[x+1,11,self.seq_len-x],1)[1] ,tf.range(self.seq_len),dtype=tf.float32),[1,0,2,3]),[self.num_seqs*self.seq_len,11,self.input_feature_size]),3))
       for i in range(len(self.conv_filter_shapes)):
            self.conv_2dlayer(i)
       self.reshape_layer()
       if self.attention_number==0:
           self.fw_cell=self.cell_create_norm()
           self.weight_bias_init()
       elif self.attention_number>0:
           self.fw_cell=self.cell_create('1')
           self.fw_cell2=self.cell_create('2')
           self.weight_bias_init()
     
       if self.attention_number==0:
           self.bw_cell=self.cell_create_norm()
           self.outputs, self.states= tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.fc_layer_out,
                                     sequence_length=self.seq,dtype=tf.float32)
    
    
           self.presoft=tf.map_fn(lambda x:tf.matmul(x,self.weights)+self.biases,tf.concat((self.outputs[0],self.outputs[1]),2))
           
           
       elif self.attention_number>0:
           self.bw_cell=self.cell_create('1')
           self.bw_cell2=self.cell_create('2') 
           with tf.variable_scope('1'):
    
               self.outputs, self.states= tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.fc_layer_out,
                                                 sequence_length=self.seq,dtype=tf.float32)
                                              
           self.first_out=tf.concat((self.outputs[0],self.outputs[1]),2)
           with tf.variable_scope('2'):
               self.outputs2, self.states2= tf.nn.bidirectional_dynamic_rnn(self.fw_cell2, self.bw_cell2, self.first_out,
                                                 sequence_length=self.seq,dtype=tf.float32)
           self.second_out=tf.concat((self.outputs2[0],self.outputs2[1]),2)
    
           for i in range((self.attention_number*2)+1):
               self.attention_weight_init(i)
    
           
            
           self.zero_pad_second_out=tf.map_fn(lambda x:tf.pad(tf.squeeze(x),[[self.attention_number,self.attention_number],[0,0]]),self.second_out)
           self.first_out_reshape=tf.reshape(self.first_out,[-1,self.n_hidden[self.n_layers-1]*2])
           self.zero_pad_second_out_reshape=[]
           self.attention_m=[]
           for j in range((self.attention_number*2)+1):
               self.zero_pad_second_out_reshape.append(tf.reshape(tf.slice(self.zero_pad_second_out,[0,j,0],[self.num_seqs,self.seq_len,self.n_hidden[self.n_layers-1]*2]),[-1,self.n_hidden[self.n_layers-1]*2]))
               self.attention_m.append(tf.tanh(tf.matmul(tf.concat((self.zero_pad_second_out_reshape[j],self.first_out_reshape),1),self.attention_weights[j])))
           self.attention_s=tf.nn.softmax(tf.stack([tf.matmul(self.attention_m[j],self.sm_attention_weights[j]) for j in range(self.attention_number*2+1)]),0)
           self.attention_z=tf.reduce_sum([self.attention_s[j]*self.zero_pad_second_out_reshape[j] for j in range(self.attention_number*2+1)],0)
           self.attention_z_reshape=tf.reshape(self.attention_z,[self.num_seqs,self.seq_len,self.n_hidden[self.n_layers-1]*2])
           self.presoft=tf.map_fn(lambda x:tf.matmul(x,self.weights)+self.biases,self.attention_z_reshape)
                         
       self.pred=tf.nn.sigmoid(self.presoft)    
       if self.cost_type=='CE':
           self.cost_transcriber = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.presoft,[-1,self.n_classes]), labels=tf.reshape(self.y_ph,[-1,self.n_classes])))
           self.cost_player = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.presoft,[-1,self.n_classes]), labels=1-tf.reshape(self.y_ph,[-1,self.n_classes])))

       ## multiple time step loss function proposed in [2]
       elif self.cost_type=='WMD':          
           self.cost_transcriber = WMD(self.pred,self.y_ph,1/4.,self.seq_len-2,1.0) 
           self.cost_player = WMD(self.pred,1-self.y_ph,1/4.,self.seq_len-2,1.0)

        # trainable variables for each network
       self.t_vars = tf.trainable_variables()
       self.player_vars=[]
       self.transcriber_vars=[]
       for var in self.t_vars:
           if 'player_' in var.name:
              self.player_vars.append(var)
           else:
               self.transcriber_vars.append(var)           
               
    #Adam optimiser, optimise player using only player parameters and transcriber using only transcriber parameters
       self.optimize_player= tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_player,var_list=self.player_vars)
       self.optimize_transcriber = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_transcriber,var_list=self.transcriber_vars)

       self.init = tf.global_variables_initializer()
       self.saver = tf.train.Saver()
       self.saver_var = tf.train.Saver(tf.trainable_variables())
       if self.save_location==[]:
           self.save_location=os.getcwd()

     def train(self):
        
       self.iteration=0
       self.epoch=0
       self.prev_val_loss=100
       self.val_loss=99
       config=tf.ConfigProto(intra_op_parallelism_threads=20)
       with tf.Session(config=config) as sess:
             sess.run(self.init)             
             while self.epoch < self.minimum_epoch or self.prev_val_loss > self.val_loss:

                 for i in range(self.num_batch):
                     R=[]
                     for k in range(3):
                         R.append(np.random.permutation(len(self.sample_specs[k])))
                     self.snippets_in=[]
                     for k in range(self.num_samples_per_batch):
                         self.snippets_in.append([])
                         self.snippets_in[k].append(self.sample_specs[0][R[0][k]])
                         self.snippets_in[k].append(self.sample_specs[1][R[1][k]])
                         self.snippets_in[k].append(self.sample_specs[2][R[2][k]])
                         self.snippets_in[k]=np.array(self.snippets_in[k])
                     if self.player_on=='yes':
                         sess.run(self.optimize_player, feed_dict={self.snippets_existing:np.expand_dims(np.reshape(np.expand_dims(self.features[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.input_feature_size]),0),self.snippets_samples: np.array(self.snippets_in), self.targets: np.reshape(np.expand_dims(self.targ[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.n_classes]),self.dropout_ph:1,self.seq:np.ones(int(self.batch_size/self.snippet_length))*self.snippet_length,self.num_seqs:10, self.seq_len:self.snippet_length,self.sample_tv_switch:0})

                     sess.run(self.optimize_transcriber, feed_dict={self.snippets_existing: np.expand_dims(np.reshape(np.expand_dims(self.features[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.input_feature_size]),0), self.snippets_samples: np.array(self.snippets_in),self.targets: np.reshape(np.expand_dims(self.targ[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.n_classes]),self.dropout_ph:1,self.seq:np.ones(int(self.batch_size/self.snippet_length))*self.snippet_length,self.num_seqs:10, self.seq_len:self.snippet_length,self.sample_tv_switch:0})
#                     
                 print("Epoch " + str(self.epoch))

                 self.epoch+=1  
                 if self.epoch > self.minimum_epoch:
                     self.loss_val_transcriber=[]
                     self.loss_val_player=[]                     
                     for i in range(self.val_num_batch):
#                         print('val started')
                         self.loss_val_transcriber.append(sess.run(self.cost_transcriber, feed_dict={self.snippets_existing: np.expand_dims(np.expand_dims(self.val[i*self.batch_size:(i+1)*self.batch_size,:],0),0),self.snippets_samples: np.array(self.snippets_in), self.targets: np.expand_dims(self.val_targ[i*self.batch_size:(i+1)*self.batch_size,:],0).astype(float),self.dropout_ph:1,self.seq:[self.batch_size],self.num_seqs:1, self.seq_len:self.batch_size,self.sample_tv_switch:1}))
#                         self.loss_val_player.append(sess.run(self.cost_player, feed_dict={self.snippets: np.expand_dims(np.expand_dims(self.val[i*self.batch_size:(i+1)*self.batch_size,:],0),0), self.targets: np.expand_dims(self.val_targ[i*self.batch_size:(i+1)*self.batch_size,:],0).astype(float),self.dropout_ph:1,self.seq:[self.batch_size],self.num_seqs:1, self.seq_len:self.batch_size,self.sample_tv_switch:1,  self.aug:np.random.rand(self.num_samples_per_batch,2,3,self.sample_len,self.input_feature_size)}))
#                                                  
                     self.prev_val_loss=self.val_loss
                     self.val_loss=np.mean(np.array(self.loss_val_transcriber))              
                     print("Val Minibatch Loss_Transcriber " + "{:.6f}".format(self.val_loss))
                 if self.epoch>=self.maximum_epoch:
                  break
             print("Optimization Finished!")
             
             ## plot stages of player model
             self.input_data,self.sample_inputs,self.input_target,self.input_data_aug,self.sample_inputs_aug1,self.generated_locations,self.pos_locations,self.sample_locs,self.samples_in_locations,self.new_data, self.new_targets=sess.run((self.snippets1,self.snippets_samples1,self.locations3,self.snippets_p_exist,self.snippets_p1,self.locations13,self.loc_valsf1,self.locations14,self.spectrums1,self.overall_spectrum,self.locations_overall), feed_dict={self.snippets_existing:np.expand_dims(np.reshape(np.expand_dims(self.features[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.input_feature_size]),0),self.snippets_samples: np.array(self.snippets_in), self.targets: np.reshape(np.expand_dims(self.targ[i*self.batch_size:(i+1)*self.batch_size,:],0),[-1,self.snippet_length,self.n_classes]),self.dropout_ph:1,self.seq:np.ones(int(self.batch_size/self.snippet_length))*self.snippet_length,self.num_seqs:int(self.batch_size/float(self.snippet_length)), self.seq_len:self.snippet_length,self.sample_tv_switch:0})
             plt.close('all')
             plt.figure(facecolor='white')
             title_font_size=10
             ax=plt.subplot(5,5,1)
             plt.axis('off')
             plt.imshow(np.transpose(self.input_data[0,0]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Existing Training Data",fontsize=title_font_size)
             ax=plt.subplot(5,5,2)
             plt.axis('off')
             plt.plot(self.input_target[0,:,0])
             ax.set_title("Existing Training Data Kick Drum Target",fontsize=title_font_size)
             ax=plt.subplot(5,5,3)
             plt.axis('off')
             plt.plot(self.input_target[0,:,1]) 
             ax.set_title("Existing Training Data Snare Drum Target",fontsize=title_font_size)
             ax=plt.subplot(5,5,4)
             plt.axis('off')
             plt.plot(self.input_target[0,:,2])
             ax.set_title("Existing Training Data Hihat Target",fontsize=title_font_size)             
             ax=plt.subplot(5,5,5)
             plt.axis('off')
             plt.imshow(np.transpose(self.sample_inputs[0,0]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Kick Drum Sample",fontsize=title_font_size)
             ax=plt.subplot(5,5,6)
             plt.axis('off')
             plt.imshow(np.transpose(self.sample_inputs[0,1]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Snare Drum Sample",fontsize=title_font_size)
             ax=plt.subplot(5,5,7)
             plt.axis('off')
             plt.imshow(np.transpose(self.sample_inputs[0,2]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Hihat Sample",fontsize=title_font_size)
             ax=plt.subplot(5,5,8)
             plt.axis('off')
             plt.imshow(np.transpose(self.input_data_aug[0]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Augmented Existing Training Data",fontsize=title_font_size)
             ax=plt.subplot(5,5,9)
             plt.axis('off')
             plt.imshow(np.transpose(self.sample_inputs_aug1[0,0]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Augmented Kick Drum Sample",fontsize=title_font_size)
             ax=plt.subplot(5,5,10)
             plt.axis('off')
             plt.imshow(np.transpose(self.sample_inputs_aug1[0,1]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Augmented Snare Drum Sample",fontsize=title_font_size)
             ax=plt.subplot(5,5,11)
             plt.axis('off')
             plt.imshow(np.transpose(self.sample_inputs_aug1[0,2]),aspect='auto',cmap='gray_r',origin='lower')             
             ax.set_title("Augmented Hihat Sample",fontsize=title_font_size)
             ax=plt.subplot(5,5,12)
             plt.axis('off')
             plt.plot(self.generated_locations[0,:,0])
             ax.set_title("Generated Kick Drum Location",fontsize=title_font_size)
             ax=plt.subplot(5,5,13)
             plt.axis('off')
             plt.plot(self.generated_locations[0,:,1])
             ax.set_title("Generated Snare Drum Location",fontsize=title_font_size)
             ax=plt.subplot(5,5,14)
             plt.axis('off')
             plt.plot(self.generated_locations[0,:,2])
             ax.set_title("Generated Hihat Location",fontsize=title_font_size) 
             ax=plt.subplot(5,5,15)
             plt.axis('off')
             plt.plot(self.pos_locations[0,:,0])
             ax.set_title("Possible Kick Drum Locations",fontsize=title_font_size)
             ax=plt.subplot(5,5,16)
             plt.axis('off')
             plt.plot(self.pos_locations[0,:,1])
             ax.set_title("Possible Snare Drum Locations",fontsize=title_font_size)
             ax=plt.subplot(5,5,17)
             plt.axis('off')
             plt.plot(self.pos_locations[0,:,2])
             ax.set_title("Possible Hihat Locations",fontsize=title_font_size) 
             ax=plt.subplot(5,5,18)
             plt.axis('off')
             plt.plot(self.sample_locs[0,:,0])
             ax.set_title("Kick Drum Location",fontsize=title_font_size)
             ax=plt.subplot(5,5,19)
             plt.axis('off')
             plt.plot(self.sample_locs[0,:,1])
             ax.set_title("Snare Drum Location",fontsize=title_font_size)
             ax=plt.subplot(5,5,20)
             plt.axis('off')
             plt.plot(self.sample_locs[0,:,2])
             ax.set_title("Hihat Location",fontsize=title_font_size)
             ax=plt.subplot(5,5,21)
             plt.axis('off')
             plt.imshow(np.transpose(self.samples_in_locations[0]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Samples In Locations",fontsize=title_font_size)
             ax=plt.subplot(5,5,22)
             plt.axis('off')
             plt.imshow(np.transpose(self.new_data[0]),aspect='auto',cmap='gray_r',origin='lower')
             ax.set_title("Samples Added To Existing Data / New Data",fontsize=title_font_size)
             ax=plt.subplot(5,5,23)
             plt.axis('off')
             plt.plot(self.new_targets[0,:,0])
             ax.set_title("New Kick Drum Target",fontsize=title_font_size)
             ax=plt.subplot(5,5,24)
             plt.axis('off')
             plt.plot(self.new_targets[0,:,1])
             ax.set_title("New Snare Drum Target",fontsize=title_font_size)
             ax=plt.subplot(5,5,25)
             plt.axis('off')
             plt.plot(self.new_targets[0,:,2])
             ax.set_title("New Hihat Target",fontsize=title_font_size)             
             self.saver.save(sess, self.save_location+'/'+self.filename)

         
     def implement(self,data):
             self.data=data
             self.test_out=[]
             with tf.Session() as sess:
                     self.saver.restore(sess, self.save_location+'/'+self.filename)
                     for i in range(len(self.data)):
                             self.test_out.append(sess.run(self.pred, feed_dict={self.snippets_existing: np.expand_dims(np.expand_dims(self.data[i],0),0),self.snippets_samples: np.array(self.snippets_in),self.targets: np.random.randn(10,100,3),self.dropout_ph:1,self.seq:[len(self.data[i])],self.num_seqs:1, self.seq_len:len(self.data[i]),self.sample_tv_switch:1})[0]) 
             return self.test_out         
         
    
    
###########

#################################################################################
             
## Example Implementation
             
# load logarithmic spectrograms
TrainSpec=np.load('ExampleTrainSpec.npy')
TrainTarg=np.load('ExampleTrainTarg.npy')
ValSpec=np.load('ExampleValSpec.npy')
ValTarg=np.load('ExampleValTarg.npy')
TestSpec=np.load('ExampleTestSpec.npy')
TestTarg=np.load('ExampleTestTarg.npy')
Samples=np.load('ExampleSamples.npy')

## train the network and process test data

NN=CNNSA3F5_AugAddExist(TrainSpec, TrainTarg, ValSpec, ValTarg, 'AugAddExist_test', minimum_epoch=100, maximum_epoch=200, n_hidden=[20,20], n_classes=3, attention_number=3, dropout=0.75, learning_rate=0.0003 ,save_location=[],snippet_length=100,cost_type='WMD',batch_size=1000,input_feature_size=84,conv_filter_shapes=[[3,3,1,32],[3,3,32,64]], conv_strides=[[1,1,1,1],[1,1,1,1]], pool_window_sizes=[[1,3,3,1],[1,3,3,1]],
                  sample_spec_path='ExampleSamples.npy',sample_aug_val=0.0001,sample_num_locations=75,no_sample_ins=3,sample_num_batch=1,eps=np.finfo(np.float32).eps,aug_amp_min=0.5,peak_distance=3,player_conv_filter_shapes=[[3,3,1,5],[3,3,5,10]],player_pool_window_sizes=[[1,8,7,1],[1,7,7,1]],player_conv_strides=[[1,1,1,1],[1,1,1,1]],player_on='yes')                            

TrainSpec=[]
TrainTarg=[]
ValSpec=[]
ValTarg=[]
Sample=[]
NN.create()
NN.train()
out=NN.implement([TestSpec])

## plot the output
plt.figure(facecolor='white')
ax=plt.subplot(2,1,1)
plt.plot(out[0][:,0])
ax.set_title("PvT Output")
ax=plt.subplot(2,1,2)
plt.plot(TestTarg[:,0])
ax.set_title("Ground Truth / Target")