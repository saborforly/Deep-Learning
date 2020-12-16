from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#batch_size 一次对128个图片进行训练，不能使用单独的图片去验证
from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf

import cifar10
import cifar10_input
from tensorflow.python.framework.errors_impl import InvalidArgumentError

import argparse

parser = argparse.ArgumentParser(description='Run Evaluation.')
parser.add_argument("--input", default=r'G:/python/assignment2/cs231n/cifar10_data/kitten.jpg',  help="Directory where to read model input.")
parser.add_argument("--output", default="data.txt", help="Output file name")
parser.add_argument("--ckpdir", default="G:/python/assignment2/cs231n/train", help="Directory where to read model checkpoints.")
args = parser.parse_args()

def eval_once(saver, evals,top_k_op, output):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(args.ckpdir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/junonn_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
            labels,logits = sess.run(evals)
            result = np.concatenate([labels,logits],axis=1)
            print(top_k_op)
            np.savetxt(output,result,'%s')


        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        input=args.input
        from PIL import Image
        #im=Image.open(input)
        image = tf.image.decode_jpeg(input)
        image = tf.image.resize_images(image, [24,24],method=0) 
        image = tf.cast(image, tf.float32)
       
        image=tf.expand_dims(image,0)
        print(image.shape,image.dtype)
       
        
        
        
       
        #images, labels = cifar10_input.inputs(eval_data=eval_data,False,1)
    # Build a Graph that computes the logits predictions from the
    # inference model.
        logits = cifar10.inference(image)
    # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        
        evals= [labels,logits]

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
        junonn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        output = file(args.output,'w')
        eval_once(saver, evals,top_k_op, output)
        output.close()


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
