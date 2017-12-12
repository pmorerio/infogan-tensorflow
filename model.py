import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import lrelu

class infogan(object):

    def __init__(self, mode='train',noise_dim=50,n_cat_codes=0,n_cont_codes=0,
		    learning_rate=0.0001, lambda_cat=1., lambda_cont=0.1):
        self.mode = mode
        self.learning_rate = learning_rate 
	self.n_cont_codes = n_cont_codes
	self.n_cat_codes = n_cat_codes
	self.noise_dim = noise_dim
	self.lambda_cat = lambda_cat
	self.lambda_cont = lambda_cont 
	    
    def G(self, inputs, cat_codes, cont_codes, reuse=False):
	
	if cat_codes is not None: 
	    inputs = tf.concat([inputs,cat_codes], axis=-1)
	if cont_codes is not None:
	    inputs = tf.concat([inputs,cont_codes], axis=-1)
	
	# make input a virtual 1x1 image (batch_size, 1, 1, noise+cat+cont)
	if inputs.get_shape()[1] != 1:
	    inputs = tf.expand_dims(inputs, 1)
	    inputs = tf.expand_dims(inputs, 1)
	
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

		    net = slim.conv2d(inputs,1024,[1,1],scope='fc1')#FC1
		    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 128, [7, 7], padding='VALID', scope='conv_transpose1')   # (batch_size, 7, 7, x)
                    net = slim.batch_norm(net, scope='bn_conv_transpose1')
                    net = slim.conv2d_transpose(net, 64, [4, 4], scope='conv_transpose2')  # (batch_size, 14, 14, x)
                    net = slim.batch_norm(net, scope='bn_conv_transpose2')
		    net = slim.conv2d_transpose(net, 1, [4, 4], activation_fn=tf.nn.tanh, scope='conv_transpose3')  # (batch_size, 28, 28, 1)
		    return net
	    
	    
    def D(self, images, reuse=False): # D has actually two heads D & Q
	
        # images: (batch, 28 28, 1)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=lrelu, is_training=(self.mode=='train')):
                    
                    net = slim.conv2d(images, 64, [4, 4], stride=2, scope='conv1')   
                    #~ net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 128, [4, 4],stride=2, scope='conv2')  
                    net = slim.batch_norm(net, scope='bn2') 
		    net=slim.flatten(net)
                    net = slim.fully_connected(net, 1024, scope='fc1') 
		    net = slim.batch_norm(net, scope='bn_fc1')
		    G = slim.fully_connected(net,1, activation_fn=tf.sigmoid, scope='sigmoid') 

		    dim_q = self.n_cat_codes+ 2* self.n_cont_codes
		    if dim_q==0:
			return G, None
		    else:
			Q = slim.fully_connected(net,128, scope='q_fc1')
			Q = slim.batch_norm(Q, scope='bn_q_fc1')
			Q = slim.fully_connected(Q,dim_q ,activation_fn=None, scope='q_out') 
			return G, Q


    def build_model(self):
        
	if self.mode == 'train':
	    print('Building model')
	    
	    # Placeholders for noise, codes and images
	    
	    self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
	    if self.n_cat_codes > 0:
		self.cat_codes = tf.placeholder(tf.float32, [None, self.n_cat_codes], 'cat_codes')
		#~ self.lambda_cat_ph = tf.placeholder(tf.float32, name='lambda_cat_codes')
		#~ lambda_cat_summary = tf.summary.scalar('lambda_cat', self.lambda_cat_ph)
	    else:
		self.cat_codes = None
		
	    if self.n_cont_codes > 0:
		self.cont_codes = tf.placeholder(tf.float32, [None, self.n_cont_codes], 'cont_codes')
	    else:
		self.cont_codes = None
		
            self.images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'mnist_images')
	    
	    self.fake_images = self.G(self.noise, self.cat_codes, self.cont_codes)
	    
	    self.logits_real, _ = self.D(self.images) # too bad Q is not used for real samples...
	    self.logits_fake, self.Q_logits = self.D(self.fake_images, reuse=True)
	    
	    if self.n_cat_codes > 0:
		self.Q_loss_cat = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
						labels=self.cat_codes, 
						logits=self.Q_logits))
		self.pred = tf.argmax(tf.nn.softmax(self.Q_logits),1)
		self.correct_prediction = tf.equal(self.pred, tf.argmax(self.cat_codes,1))
		#correct_prediction = tf.equal(tf.nn.top_k(y_conv,2)[1], tf.nn.top_k(y_,2)[1])
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		
		Q_loss_cat_summary = tf.summary.scalar('Q_loss_cat', self.Q_loss_cat)
		acc_summary = tf.summary.scalar('acc_cat', self.accuracy)
		

	    else:
		self.Q_loss_cat = 0
	    
	    # Losses
	    
	    self.D_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.D_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.D_loss = self.D_loss_real + self.D_loss_fake 
	    self.G_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_real)))
	    
	    # Optimizers (DC-GAN paper says momentum=0.5)
	    
            self.D_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) 
            self.G_optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=0.5)
            
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]
            
            
	    # Train ops
	    
            with tf.variable_scope('training_op',reuse=False):
                self.D_train_op = slim.learning.create_train_op(self.D_loss, self.D_optimizer, variables_to_train=D_vars)
		self.G_train_op = slim.learning.create_train_op(self.G_loss + self.lambda_cat* self.Q_loss_cat, self.G_optimizer, variables_to_train=G_vars)
            
            
            # Summary ops
	    
            G_loss_summary = tf.summary.scalar('G_loss', self.G_loss)
            D_loss_summary = tf.summary.scalar('D_loss', self.D_loss)
            D_loss_real_summary = tf.summary.scalar('D_loss_real', self.D_loss_real)
            D_loss_fake_summary = tf.summary.scalar('D_loss_fake', self.D_loss_fake)
	    gen_images_summary = tf.summary.image('gen_images', self.fake_images,max_outputs=6)
            self.summary_op = tf.summary.merge_all()
            #~ self.summary_op = tf.summary.merge([G_loss_summary, 
						#~ D_loss_summary, 
						#~ D_loss_fake_summary, 
						#~ D_loss_real_summary,
						#~ gen_images_summary])
            

            #~ for var in tf.trainable_variables():
		#~ tf.summary.histogram(var.op.name, var) 
	
	elif self.mode == 'test':
	    print('Testing model')
	    
	     # Placeholders for noise, codes and images
	    
	    self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
	    if self.n_cat_codes > 0:
		self.cat_codes = tf.placeholder(tf.float32, [None, self.n_cat_codes], 'cat_codes')
	    else:
		self.cat_codes = None
		
	    if self.n_cont_codes > 0:
		self.cont_codes = tf.placeholder(tf.float32, [None, self.n_cont_codes], 'cont_codes')
	    else:
		self.cont_codes = None
			    
	    self.fake_images = self.G(self.noise, self.cat_codes, self.cont_codes)
	    gen_images_summary = tf.summary.image('gen_images', self.fake_images,max_outputs=10)
	    
	    self.logits_fake, self.Q_logits = self.D(self.fake_images)
	    
	    self.pred = tf.argmax(tf.nn.softmax(self.Q_logits),1)
	    self.correct_prediction = tf.equal(self.pred, tf.argmax(self.cat_codes,1))
	    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
	    acc_summary = tf.summary.scalar('acc_cat', self.accuracy)
	    
	    
	    self.summary_op = tf.summary.merge_all()

	else:
	    print('Unrecognized mode')
	
	
	
if __name__=='__main__':

    model = infogan()
    model.build_model()
