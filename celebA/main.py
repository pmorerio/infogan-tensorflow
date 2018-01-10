import tensorflow as tf
from model import attrGAN
from solver import Solver



flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "'GPU id'")
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('cat_codes', '0', "number of categorcial codes")
flags.DEFINE_string('lambda_cat', '0.1', "number of categorcial codes")
flags.DEFINE_string('batch_size', '32', "batch size")
FLAGS = flags.FLAGS

def main(_):
    
    with tf.device('/gpu:'+FLAGS.gpu):

	model = attrGAN(mode=FLAGS.mode, n_cat_codes=int(FLAGS.cat_codes), 
			lambda_cat=float(FLAGS.lambda_cat), learning_rate=0.0001)
	solver = Solver(model, model_save_path=FLAGS.model_save_path+'/'+FLAGS.gpu, 
			batch_size=int(FLAGS.batch_size), log_dir='logs/'+FLAGS.gpu)
	
	if FLAGS.mode == 'train':
		solver.train()
	elif FLAGS.mode == 'test':
		solver.test()
	else:
	    print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()



    


