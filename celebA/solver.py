import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np
import os
import glob
from PIL import Image


import utils


class Solver(object):

    def __init__(self, model, batch_size=32,  model_save_path='model', 
		    log_dir='logs', data_dir='/data/datasets/CelebA',
		    train_iter=100000):
        
        self.model = model
        self.batch_size = batch_size
	self.model_save_path = model_save_path
	self.log_dir = log_dir
	self.data_dir = data_dir
	self.train_iter=train_iter
	self.train_filenames=glob.glob(self.data_dir+'/train*.tfrecords') #not sorted
	self.val_filenames=glob.glob(self.data_dir+'/val*.tfrecords') #not sorted
	self.test_filenames=glob.glob(self.data_dir+'/test*.tfrecords') #not sorted
	assert len(self.train_filenames) != 0
	assert len(self.val_filenames) != 0
	assert len(self.test_filenames) != 0
	
	# create directories if not exist
	if not tf.gfile.Exists(self.model_save_path):
	    tf.gfile.MakeDirs(self.model_save_path)

	self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	self.config.allow_soft_placement=True
	

    def read_and_decode(self):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(self.filename_queue)

	feat={'attributes':  tf.FixedLenFeature([40], tf.int64),
		'image': tf.FixedLenFeature([], tf.string)		
		}
	 # Decode the record read by the reader	
	features = tf.parse_single_example(serialized_example, features=feat)
	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.float32)
	attributes = tf.cast(features['attributes'], tf.float32)
	# Reshape image data into the original shape
	image = tf.reshape(image, [178/2, 218/2, 3])
	return image, attributes
	
	
    def inputs(self):
	
	filenames = glob.glob(self.data_dir+'/train*.tfrecords') #not sorted
	#~ print('Reading', filenames)
	with tf.name_scope('input'):
	    self.filename_queue = tf.train.string_input_producer(
	    filenames, num_epochs=None)

	    image, att = self.read_and_decode()
	    noise = tf.random_uniform([self.model.noise_dim])
	    #Shuffle the examples and collect them into batch_size batches.
	    # (Internally uses a RandomShuffleQueue.)
	    # We run this in two threads to avoid a bottleneck.
	    # * min_after_dequeue defines how big a buffer we will randomly sample
	    #   from -- bigger means better shuffling but slower start up and more
	    #   memory used.
	    # * capacity must be larger than min_after_dequeue and the amount larger
	    #   determines the maximum we will prefetch.  Recommendation:
	    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
	    return tf.train.shuffle_batch(
		[image, att, noise], batch_size=self.batch_size, num_threads=4,
		capacity=1000 + 5 * self.batch_size,
		min_after_dequeue=1000,
		allow_smaller_final_batch=True) 
	    
    def train(self):
	
	print 'Training...'
        
        # build a graph
        model = self.model
        model.build_model(self.inputs())

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	if not tf.gfile.Exists(self.model_save_path):
	    tf.gfile.MakeDirs(self.model_save_path)

        with tf.Session(config=self.config) as sess:
            # random initialize G and D
            tf.global_variables_initializer().run()
           
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

	    
	    print ('Start training.')
            t = 0
	    
            for step in range(self.train_iter):
		print("batch index: {}".format(batch_index))
		try:
                #~ i = step % int(self.train_data.shape[0] / self.batch_size)
		#~ start = i*self.batch_size
		#~ end = (i+1)*self.batch_size

		#~ input_noise = utils.sample_Z(self.batch_size, model.noise_dim, 'uniform')
		
		#~ if model.n_cat_codes > 0:
		    #~ input_cat = utils.sample_attributes(self.batch_size, model.n_cat_codes)
		    
		    #~ feed_dict = {model.noise: input_noise, model.cat_codes: input_cat,
				    #~ model.images: self.train_data[start:end]}
		#~ else:
		    #~ feed_dict = {model.noise: input_noise, model.images: self.train_data[start:end]}
		
		#~ if (t) % 200 == 0:
		    #~ avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    #~ avg_D_real = sess.run(model.logits_real, feed_dict)
		    #~ summary, dl, gl = sess.run([model.summary_op, model.D_loss, model.G_loss], feed_dict)
		    #~ summary_writer.add_summary(summary, t)
		    #~ print ('Step: [%d/%d] \n G_loss: [%.6f] D_loss: [%.6f]' \
			       #~ %(t, self.train_iter, gl, dl))
		    #~ print 'avg_D_fake',str(avg_D_fake.mean()),'avg_D_real',str(avg_D_real.mean())
		    
		    if (t) % 2000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'model')) 
		    
		    sess.run(model.D_train_op, feed_dict)
		    sess.run(model.G_train_op, feed_dict)
		    sess.run(model.Q_train_op, feed_dict)
		    #~ t+=1
		except:
		    print('exit')
		    break
	    # Stop the threads
	    coord.request_stop()

	    # Wait for threads to stop
	    coord.join(threads)

    def test(self, split='test'):
	
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
		print(t)
		# batch is same size as number of classes here
		batch_size = 3#model.n_cat_codes
		noise = utils.sample_Z(1, model.noise_dim, 'uniform')
		input_noise = np.concatenate([noise for i in range(batch_size)], axis=0)
			
		if model.n_cat_codes > 0:
		    #~ input_cat = np.asarray([[0,0,0,0,0,1,1,1,1,1],
					    #~ [1,1,1,1,1,1,1,1,1,1],
					    #~ [1,0,0,0,0,0,0,0,0,0],
					    #~ [0,1,0,0,0,0,0,0,0,0],
					    #~ [1,1,0,0,0,0,0,0,0,0],
					    #~ [0,0,0,0,0,0,0,0,0,0],
					    #~ [0,0,0,0,0,0,0,0,0,0.25],
					    #~ [0,0,0,0,0,0,0,0,0,.5],
					    #~ [0,0,0,0,0,0,0,0,0,0.75],
					    #~ [0,0,0,0,0,0,0,0,0,1]]).astype(float)
		    #~ input_cat = np.asarray([[1,1,1,1,1],
					    #~ [0,0,0,0,0],
					    #~ [0,0,0,0,1],
					    #~ [1,0,1,0,0],
					    #~ [1,0,1,0,1],
					    #~ ]).astype(float)
		    #~ input_cat = np.zeros((model.n_cat_codes,model.n_cat_codes))
		    #~ input_cat = np.eye(model.n_cat_codes)
		    input_cat = np.asarray([[1],[0.5], [0]]).astype(float)
		    
		    #~ print(input_cat)
		    feed_dict = {model.noise: input_noise, model.cat_codes: input_cat}
		else:
		    feed_dict = {model.noise: input_noise}
		
		summary, preds, imgs = sess.run([model.summary_op, model.pred, model.fake_images], feed_dict)
		summary_writer.add_summary(summary, t)
		print(np.round(preds))
		t+=1
		if t==1000:
		    break
	    

if __name__=='__main__':
    print('empty')

    
