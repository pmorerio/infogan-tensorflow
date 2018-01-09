from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

## probability for each pixel in mnist
class Info(object):

    def __init__(self, data_dir='/data/datasets/mnist'):
        
	self.data_dir = data_dir
	self.bins=256

    def load_data(self):
	self.mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
	#~ self.train_data = mnist.train.images.reshape(((len(mnist.train.labels),28,28,1)))#*2. - 1.
	#~ self.train_labels = mnist.train.labels
	#~ self.test_data = mnist.test.images.reshape(((len(mnist.test.labels),28,28,1)))#*2. - 1.
	#~ self.test_labels = mnist.test.labels
	#~ self.val_data = mnist.validation.images.reshape(((len(mnist.validation.labels),28,28,1)))#*2. - 1.
	#~ self.val_labels = mnist.validation.labels
	#~ mnist = None

    def run(self, threshold=0.):
	self.info = np.zeros((28*28), dtype=np.float)
	pixels = np.transpose(self.mnist.train.images)
	i=0
	for pixel_dist in pixels:
	    h = np.histogram(pixel_dist, bins=self.bins, normed=True)[0] / float(self.bins)
	    self.info[i] = - np.sum(h*np.log(h))
	    #~ print self.info[i]
	    i+=1
	
	#visualization
	test.info[np.isnan(test.info)]=0.  
	test.info[test.info<threshold]=threshold
	    
    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    test = Info()
    test.load_data()
    test.run()
    print np.min(test.info)
    print np.max(test.info)
    plt.imshow(np.reshape(test.info, (28,28)), cmap='gray')
    plt.show()
    
    #very same image
    summation = np.sum(test.mnist.train.images, axis=0)
    plt.imshow(np.reshape(summation, (28,28)), cmap='gray')
    plt.show()
