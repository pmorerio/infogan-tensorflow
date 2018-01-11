import numpy as np
import argparse
import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', action="store", dest='datadir', 
			default='/data/datasets/CelebA')
	args = parser.parse_args()
	_dir=args.datadir
	
	if not os.path.exists(_dir):
		print 'Cannot find folder .'
		print 'Usage: python create_tfRecords.py --dir=\'path_to_data_dir/\''
		return 0
		
	images2TFRecords(_dir)

	return 1


def images2TFRecords(dataDir, w=218/2, h=178/2, examples_per_tfrecord = 1000):
	
	attr_file = open(dataDir+'/Anno/list_attr_celeba.txt')
	n_images = int(attr_file.readline())
	n_attributes = len(attr_file.readline().split())
	
	paths = sorted(glob.glob(dataDir+'/Img/img_align_celeba/*'))
	assert len(paths) == n_images
	
	#~ splits=open(dataDir+'/Eval/list_eval_partition.txt')
	train_idx = 162770
	val_idx = 182637
	image_counter=0
	
	num_of_train_records = train_idx/examples_per_tfrecord + 1
	num_of_val_records =  (val_idx-train_idx)/1000 + 1
	num_of_test_records =  (n_images-val_idx)/1000 + 1
	
	## train
	writer = tf.python_io.TFRecordWriter(dataDir + '/train-001-of-'+str(num_of_train_records)  + '.tfrecords')
	t=1
	for i, path in enumerate(paths[:train_idx]):
		img = Image.open(path)
		img = img.resize((w,h), Image.ANTIALIAS)
		img = np.array(img, dtype=float) 
		filename = path.split('/')[-1]
		
		attributes = attr_file.readline().split()
		assert filename == attributes[0]
		attributes = attributes[1:]
		assert len(attributes) == 40
		attributes = [int(att) for att in attributes]
		
		example = tf.train.Example(features=tf.train.Features(
			feature={
			'attributes': _int64_feature_list(attributes),
			'image': 	_bytes_feature(tf.compat.as_bytes(img.astype(np.float32).tostring()) )						
			}
			))
		writer.write(example.SerializeToString())
		image_counter+=1
		
		if (i+1) % 1000 == 0:
			t+=1
			writer.close()
			writer = tf.python_io.TFRecordWriter(dataDir + '/train-'+'{:03d}'.format(t) +'-of-'+str(num_of_train_records)  + '.tfrecords')
	writer.close()
	
	## validation
	writer = tf.python_io.TFRecordWriter(dataDir + '/val-01-of-'+str(num_of_val_records)  + '.tfrecords')
	t=1
	for i, path in enumerate(paths[train_idx:val_idx]):
		img = Image.open(path)
		img = img.resize((w,h), Image.ANTIALIAS)
		img = np.array(img, dtype=float) 
		filename = path.split('/')[-1]
		
		attributes = attr_file.readline().split()
		assert filename == attributes[0]
		attributes = attributes[1:]
		assert len(attributes) == 40
		attributes = [int(att) for att in attributes]
		
		example = tf.train.Example(features=tf.train.Features(
			feature={
			'attributes': _int64_feature_list(attributes),
			'image': 	_bytes_feature(tf.compat.as_bytes(img.astype(np.float32).tostring()) )						
			}
			))
		writer.write(example.SerializeToString())
		image_counter+=1
		
		if (i+1) % 1000 == 0:
			t+=1
			writer.close()
			writer = tf.python_io.TFRecordWriter(dataDir + '/val-'+'{:02d}'.format(t) +'-of-'+str(num_of_val_records)  + '.tfrecords')
	writer.close()
	
	## test
	writer = tf.python_io.TFRecordWriter(dataDir + '/test-01-of-'+str(num_of_test_records)  + '.tfrecords')
	t=1
	for i, path in enumerate(paths[val_idx:]):
		img = Image.open(path)
		img = img.resize((w,h), Image.ANTIALIAS)
		img = np.array(img, dtype=float) 
		filename = path.split('/')[-1]
		
		attributes = attr_file.readline().split()
		assert filename == attributes[0]
		attributes = attributes[1:]
		assert len(attributes) == 40
		attributes = [int(att) for att in attributes]
		
		example = tf.train.Example(features=tf.train.Features(
			feature={
			'attributes': _int64_feature_list(attributes),
			'image': 	_bytes_feature(tf.compat.as_bytes(img.astype(np.float32).tostring()) )						
			}
			))
		writer.write(example.SerializeToString())
		image_counter+=1
		
		if (i+1) % 1000 == 0:
			t+=1
			writer.close()
			writer = tf.python_io.TFRecordWriter(dataDir + '/test-'+'{:02d}'.format(t) +'-of-'+str(num_of_test_records)  + '.tfrecords')
	writer.close()
		
		
		
	assert image_counter == n_images
	attr_file.close()
	

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))	
	
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == "__main__":
    main()
