from data import next_batch_, imcombind_, imsave_
from ops import *
from sampler import gaussian

flags = tf.app.flags
flags.DEFINE_string('log_path', './logs/lsgan/', '')

flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('bz', 64, '')
flags.DEFINE_integer('z_dim', 1024, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('clip_v', 0.02, '')
FLAGS = flags.FLAGS


class G(object):
    def __init__(self):
        self.name = 'mnist/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name):
            fc1 = relu(bn(dense(z, 1024)))
            fc2 = relu(bn(dense(fc1, 7 * 7 * 128)))
            fc2 = tf.reshape(fc2, [-1, 7, 7, 128])
            cv1 = relu(bn(dconv2d(fc2, 128, [5, 5], [2, 2], 'SAME')))
            fake = sigmoid(dconv2d(cv1, 1, [5, 5], [2, 2], 'SAME'))
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D(object):
    def __init__(self):
        self.name = 'mnist/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            cv1 = lrelu(conv2d(x, 11, [5, 5], [2, 2]))
            cv2 = lrelu(bn(conv2d(cv1, 74, [5, 5], [2, 2])))
            fc1 = lrelu(bn(dense(flatten(cv2), 1024)))
            # fc2 = sigmoid(dense(fc1, 1))  # 添加sigmoid太容易发生饱和，难以训练
            fc2 = dense(fc1, 1)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class ls_gan(object):
    def __init__(self):
        self.G = G()
        self.D = D()

        self.real = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
        self.fake = self.G(self.z)
        self.d_fake = self.D(self.fake, reuse=False)
        self.d_real = self.D(self.real)

        self.g_loss = tf.reduce_mean(tf.square(self.d_fake - tf.ones_like(self.d_fake)))
        self.d_loss = tf.reduce_mean(tf.square(self.d_real - tf.ones_like(self.d_real)) + tf.square(self.d_fake))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.5)
            self.d_optim = optimizer.minimize(self.d_loss, var_list=self.D.vars)
            self.g_optim = optimizer.minimize(self.g_loss, var_list=self.G.vars)

        self.fit_summary = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.histogram('d_fake', self.d_fake),
            tf.summary.histogram('d_real', self.d_real),
            tf.summary.image('X', self.real, 4),
            tf.summary.image('fake', self.fake, 4)
        ])
        self.inc_step = tf.assign_add(global_step, 1)

    def gen(self, sess, bz):
        return sess.run(self.fake, feed_dict={self.z: gaussian(bz, FLAGS.z_dim), is_training: False})

    def fit(self, sess, local_):
        for _ in range(local_):
            sess.run(self.inc_step)
            x_real, _ = next_batch_(FLAGS.bz)
            z = gaussian(FLAGS.bz, FLAGS.z_dim)
            for _ in range(3):
                sess.run(self.d_optim, feed_dict={self.real: x_real, self.z: z, is_training: True})
            sess.run(self.g_optim, feed_dict={self.real: x_real, self.z: z, is_training: True})

        x_real, _ = next_batch_(FLAGS.bz)
        return sess.run([self.d_loss, self.g_loss, self.fit_summary], feed_dict={
            self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim), is_training: False})


def main(_):
    _model = ls_gan()
    _gpu = tf.GPUOptions(allow_growth=True)
    _saver = tf.train.Saver(pad_step_number=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu)) as sess:
        _writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(FLAGS.log_path)
        if ckpt and ckpt.model_checkpoint_path:
            _saver.restore(sess, FLAGS.log_path)

        _step = global_step.eval()
        while True:
            if _step >= FLAGS.steps:
                break
            d_loss, g_loss, fit_summary = _model.fit(sess, 100)

            _step = _step + 100
            _writer.add_summary(fit_summary, _step)
            print("Train [%d\%d] g_loss [%3f] d_loss [%3f]" % (_step, FLAGS.steps, g_loss, d_loss))

            images = _model.gen(sess, 100)
            imsave_(FLAGS.log_path + 'train_{}.png'.format(_step), imcombind_(images))
            _saver.save(sess, FLAGS.log_path) if _step % 5000 == 0 else None


if __name__ == "__main__":
    tf.app.run()
