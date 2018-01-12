import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt

###############################
#Just to check tfrecords are ok
###############################


dataDir = '/data/datasets/CelebA'
train_filenames = glob.glob(dataDir+'/train*.tfrecords') #not sorted
# queue
filename_queue = tf.train.string_input_producer(train_filenames, num_epochs=1)
 # reader to read the next record
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

feat={'attributes':  tf.FixedLenFeature([40], tf.int64),
	'image': tf.FixedLenFeature([], tf.string)		
	}
 # Decode the record read by the reader	
features = tf.parse_single_example(serialized_example, features=feat)
# Convert the image data from string back to the numbers
image = tf.decode_raw(features['image'], tf.float32)
attributes = tf.cast(features['attributes'], tf.int64)
# Reshape image data into the original shape
image = tf.reshape(image, [178/2, 218/2, 3])
attributes = tf.reshape(attributes, [40])

imgs, attribs = tf.train.shuffle_batch([image, attributes], batch_size=16, capacity=30, num_threads=4,
					      min_after_dequeue=10, allow_smaller_final_batch=False)

attr_file = open(dataDir+'/Anno/list_attr_celeba.txt')
n_images = int(attr_file.readline())
att_names = attr_file.readline().split()
print att_names

with tf.Session() as sess:

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    for batch_index in range(100):
        print("batch index: {}".format(batch_index))
        try:
            img, att = sess.run([imgs, attribs])
        except:
            print('exit')
            break
        img = img.astype(np.uint8)
        for j in range(4):
            plt.subplot(4, 1, j+1)
            plt.imshow(img[j, ...])
	    a = np.asarray(att[j],dtype=int)
	    print [att_names[i] for i in np.where(a==1)[0]]
	print '----------------------'
        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
            

