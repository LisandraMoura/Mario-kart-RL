#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os, socket
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.a3c import A3C
from envs import create_env
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server, task):
    env = create_env(args.env_id, client_id=str(task), remotes=args.remotes)
    trainer = A3C(env, task, args.visualise)

    # Variable names that start with "local" are not saved in checkpoints.
    if use_tf12_api:
        variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    else:
        variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(task)])
    logdir = os.path.join(args.log_dir, 'train')

    if use_tf12_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % task)

    logger.info("Events directory: %s_%s", logdir, task)
    is_chief = (task == 0)
    logger.info("Is chief? %s", is_chief)
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def cluster_spec(is_worker, workerhost, worker_id):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}

    all_ps = []
    host = 'param_svr'
    port = 12222
    all_ps.append('{}:{}'.format(host, port))
    cluster['ps'] = all_ps

    port += 1
    if is_worker:
        cluster['worker'] = {int(worker_id): "%s:%s" % (workerhost, port)}
    else:
        all_workers = []
        worker_ips = socket.gethostbyname_ex('worker')[2]
        worker_ips.sort()
        for w in worker_ips:
            all_workers.append('{}:{}'.format(w, port))
        cluster['worker'] = all_workers
    return cluster

def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    # parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    # parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    host = socket.gethostname()
    ip=socket.gethostbyname(host)
    # TODO: Why not simply 'register' all IPs on the subnet as possible workers. Have the P.S. start as x.x.x.1 and make all others workers. That should allow for on-the-fly scaling with docker-compose
    worker_id = int(ip[ip.rfind('.') + 1:]) - 2 # HACK:
                                                # The workers should start as x.x.x.2-n
                                                # and the PS server should start as x.x.x.n+1,

    args = parser.parse_args()
    spec = cluster_spec(args.job_name == "worker", ip, worker_id)
    logger.info("Defining cluster spec as: %s", spec)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=worker_id,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server, worker_id)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=0,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
