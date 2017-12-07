import tensorflow as tf
from model import infogan
from solver import Solver



flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "'GPU id'")
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('cat_codes', '0', "number of categorcial codes")
FLAGS = flags.FLAGS

def main(_):
    
    with tf.device('/gpu:'+FLAGS.gpu):

	model = infogan(mode=FLAGS.mode, n_cat_codes=int(FLAGS.cat_codes), learning_rate=0.0005)
	solver = Solver(model, model_save_path=FLAGS.model_save_path)
	
	if FLAGS.mode == 'train':
		solver.train()
	else:
	    print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()



    


