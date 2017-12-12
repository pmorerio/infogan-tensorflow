import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os

import utils


class Solver(object):

    def __init__(self, model, batch_size=32,  model_save_path='model', 
		    log_dir='logs', data_dir='/data/datasets/mnist',
		    train_iter=500000):
        
        self.model = model
        self.batch_size = batch_size
	self.model_save_path = model_save_path
	self.log_dir = log_dir
	self.data_dir = data_dir
	self.train_iter=train_iter
	
	# create directories if not exist
	if not tf.gfile.Exists(self.model_save_path):
	    tf.gfile.MakeDirs(self.model_save_path)

	self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	self.config.allow_soft_placement=True

    def load_data(self):
	#original data is in [0:1], rescale to [-1,1]
	mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
	self.train_data = mnist.train.images.reshape(((len(mnist.train.labels),28,28,1)))*2. - 1.
	self.train_labels = mnist.train.labels
	self.test_data = mnist.test.images.reshape(((len(mnist.test.labels),28,28,1)))*2. - 1.
	self.test_labels = mnist.test.labels
	self.val_data = mnist.validation.images.reshape(((len(mnist.validation.labels),28,28,1)))*2. - 1.
	self.val_labels = mnist.train.labels
	mnist = None
	    
	    
    def train(self):
	
	print 'Training...'
        # load mnist dataset
	self.load_data()
        
        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	if not tf.gfile.Exists(self.model_save_path):
	    tf.gfile.MakeDirs(self.model_save_path)
	
	def _lambda_cat(time, gamma=1e-5):
            return model.lambda_cat * (1. - np.exp(-gamma*time))


        with tf.Session(config=self.config) as sess:
            # random initialize G and D
            tf.global_variables_initializer().run()
           
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

	    
	    print ('Start training.')
            t = 0
	    
            for step in range(self.train_iter):

                i = step % int(self.train_data.shape[0] / self.batch_size)
		start = i*self.batch_size
		end = (i+1)*self.batch_size

		input_noise = utils.sample_Z(self.batch_size, model.noise_dim, 'uniform')
		
		if model.n_cat_codes > 0:
		    input_cat = utils.sample_cat(self.batch_size, model.n_cat_codes)
		    
		    feed_dict = {model.noise: input_noise, model.cat_codes: input_cat,
				    model.images: self.train_data[start:end]}
				    #~ model.lambda_cat_ph: _lambda_cat(t)}
		else:
		    feed_dict = {model.noise: input_noise, model.images: self.train_data[start:end]}
		
		if (t) % 100 == 0:
		    avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    avg_D_real = sess.run(model.logits_real, feed_dict)
		    summary, dl, gl = sess.run([model.summary_op, model.D_loss, model.G_loss], feed_dict)
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d] \n G_loss: [%.6f] D_loss: [%.6f]' \
			       %(t, self.train_iter, gl, dl))
		    print 'avg_D_fake',str(avg_D_fake.mean()),'avg_D_real',str(avg_D_real.mean())
		    
		if (t) % 1000 == 0:  
		    saver.save(sess, os.path.join(self.model_save_path, 'model')) 
		
		sess.run(model.D_train_op, feed_dict)
		sess.run(model.G_train_op, feed_dict)
		t+=1

    def test(self):
	
	print 'Testing...'
                
	if tf.gfile.Exists(self.log_dir+'/test'):
            tf.gfile.DeleteRecursively(self.log_dir+'/test')
        tf.gfile.MakeDirs(self.log_dir+'/test')
	if not tf.gfile.Exists(self.model_save_path):
	    print('No model to test')
	    return -1
	    
        # build a graph
        model = self.model
        model.build_model()
	
	with tf.Session(config=self.config) as sess:
	    
	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir+'/test', graph=tf.get_default_graph())
	    
	    print ('Loading test model.')
	    variables_to_restore = slim.get_model_variables()
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.model_save_path+'/model')
	    print ('Done!')
	
	    t=0
	    while(True):
		# batch is same size as number of classes here
		batch_size = model.n_cat_codes
		input_noise = utils.sample_Z(batch_size, model.noise_dim, 'uniform')
			
		if model.n_cat_codes > 0:
		    input_cat = np.eye(model.n_cat_codes)
		    feed_dict = {model.noise: input_noise, model.cat_codes: input_cat}
		else:
		    feed_dict = {model.noise: input_noise}
		
		summary, preds, imgs = sess.run([model.summary_op, model.pred, model.fake_images], feed_dict)
		summary_writer.add_summary(summary,t)
		print(preds)
		t+=1
	    

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    from model import infogan
    model = infogan()
    solver = Solver(model)
    solver.load_data()
    n=9999
    print(solver.test_labels[n])
    plt.imshow(np.squeeze(solver.test_data[n]))
    plt.show()
    
