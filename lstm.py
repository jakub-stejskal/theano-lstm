# Get the train/test data from:
# https://www.dropbox.com/s/iw1uulh3h5icjuj/anbncn_1_10.nc
# https://dl.dropboxusercontent.com/u/2274024/anbncn_10_100.nc
import pickle
from scipy.io import netcdf
import numpy as np
import theano
import theano.tensor as TT
import time
import sys


def load_ncdata(fname):
    """Load NetCDF file"""
    nc = netcdf.netcdf_file(fname, 'r')
    patternSize = nc.dimensions['inputPattSize']
    numSeqs = nc.dimensions['numSeqs']

    # Alloc memory
    L = [nc.variables['seqLengths'][seq] for seq in range(numSeqs)]
    maxLength = max(L)
    X = np.zeros((maxLength, numSeqs, patternSize), dtype=np.float32)
    Y = np.zeros((maxLength, numSeqs, patternSize), dtype=np.float32)
    M = np.zeros((maxLength, numSeqs, patternSize), dtype=np.int8)
    i = 0
    for seq in range(numSeqs):
        l = nc.variables['seqLengths'][seq]
        x = nc.variables['inputs'][i:i + l]
        y = nc.variables['targetPatterns'][i:i + l]
        X[0:l, seq, :] = x
        Y[0:l, seq, :] = y
        M[0:l, seq, :] = 1
        i += l
    return X, Y, M


class LSTM_RNN:

    def __init__(self, in_dim=4, lstm_dim=5, out_dim=4, prange=0.01, parameters_filename=None,
                 learning_rate=0.001, momentum=0.9, max_epochs=50, max_time=60000, batch_size=1, seed=None):
        self.in_dim = in_dim
        self.lstm_dim = lstm_dim
        self.out_dim = out_dim
        self.prange = prange

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.max_time = max_time
        self.batch_size = batch_size

        if parameters_filename:
            self.init_state, self.lstm_params, self.lstm_prev_updates = self.load_model(parameters_filename)
        else:
            self.init_state, self.lstm_params, self.lstm_prev_updates = self.init_model()

        if seed:
            np.random.seed(seed)

    def _init_init_states(self):
        # c0_array = np.random.randn(lstm_dim)
        # Initial states
        c0_array = np.zeros((self.lstm_dim,))
        c0 = theano.shared(c0_array, name='c0')
        h0 = theano.shared(c0_array, name='h0')
        return c0, h0

    def _init_parameters(self):
        # LSTM LAYER
        # Weights
        # Input to LSTM (inp_dim x lstm_dim x 4)
        W_xi = theano.shared(np.random.randn(self.in_dim, self.lstm_dim) * self.prange, name='W_xi')
        W_xo = theano.shared(np.random.randn(self.in_dim, self.lstm_dim) * self.prange, name='W_xo')
        W_xf = theano.shared(np.random.randn(self.in_dim, self.lstm_dim) * self.prange, name='W_xf')
        W_xc = theano.shared(np.random.randn(self.in_dim, self.lstm_dim) * self.prange, name='W_xc')

        # LSTM to LSTM (self.lstm_dim x self.lstm_dim x 4)
        W_hi = theano.shared(np.random.randn(self.lstm_dim, self.lstm_dim) * self.prange, name='W_hi')
        W_ho = theano.shared(np.random.randn(self.lstm_dim, self.lstm_dim) * self.prange, name='W_ho')
        W_hf = theano.shared(np.random.randn(self.lstm_dim, self.lstm_dim) * self.prange, name='W_hf')
        W_hc = theano.shared(np.random.randn(self.lstm_dim, self.lstm_dim) * self.prange, name='W_hc')

        # Peephole (self.lstm_dim x 3)
        W_ci = theano.shared(np.random.randn(self.lstm_dim, ) * self.prange, name='W_ci')
        W_co = theano.shared(np.random.randn(self.lstm_dim, ) * self.prange, name='W_co')
        W_cf = theano.shared(np.random.randn(self.lstm_dim, ) * self.prange, name='W_cf')

        # Bias (self.lstm_dim x 4)
        b_i = theano.shared(np.random.randn(self.lstm_dim) * self.prange, name='b_i')
        b_o = theano.shared(np.random.randn(self.lstm_dim) * self.prange, name='b_o')
        b_f = theano.shared(np.random.randn(self.lstm_dim) * self.prange, name='b_f')
        b_c = theano.shared(np.random.randn(self.lstm_dim) * self.prange, name='b_c')

        # OUTPUT LAYER
        # Output weights (20) and bias (4)
        W_hy = theano.shared(np.random.randn(self.lstm_dim, self.out_dim) * self.prange, name='W_h_y')
        b_y = theano.shared(np.random.randn(self.out_dim, ) * self.prange, name='B_y')

        return (W_xi, W_xo, W_xf, W_xc, W_hi, W_ho, W_hf, W_hc,
                W_ci, W_co, W_cf, W_hy, b_i, b_o, b_f, b_c, b_y)
    
    def _init_gradients(self):
        # LSTM LAYER
        # Initial weight updates (initialized to 0)
        g_W_xi = theano.shared(np.zeros((self.in_dim, self.lstm_dim)), name='g_W_xi')
        g_W_xo = theano.shared(np.zeros((self.in_dim, self.lstm_dim)), name='g_W_xo')
        g_W_xf = theano.shared(np.zeros((self.in_dim, self.lstm_dim)), name='g_W_xf')
        g_W_xc = theano.shared(np.zeros((self.in_dim, self.lstm_dim)), name='g_W_xc')

        g_W_hi = theano.shared(np.zeros((self.lstm_dim, self.lstm_dim)), name='g_W_hi')
        g_W_ho = theano.shared(np.zeros((self.lstm_dim, self.lstm_dim)), name='g_W_ho')
        g_W_hf = theano.shared(np.zeros((self.lstm_dim, self.lstm_dim)), name='g_W_hf')
        g_W_hc = theano.shared(np.zeros((self.lstm_dim, self.lstm_dim)), name='g_W_hc')

        g_W_ci = theano.shared(np.zeros((self.lstm_dim,)), name='g_W_ci')
        g_W_co = theano.shared(np.zeros((self.lstm_dim,)), name='g_W_co')
        g_W_cf = theano.shared(np.zeros((self.lstm_dim,)), name='g_W_cf')

        # Initial Bias updates
        g_b_i = theano.shared(np.zeros((self.lstm_dim,)), name='g_b_i')
        g_b_o = theano.shared(np.zeros((self.lstm_dim,)), name='g_b_o')
        g_b_f = theano.shared(np.zeros((self.lstm_dim,)), name='g_b_f')
        g_b_c = theano.shared(np.zeros((self.lstm_dim,)), name='g_b_cNN(training_input.shape[1], 16, 0, ')

        # OUTPUT LAYER
        # Output initial gradients
        g_W_hy = theano.shared(np.zeros((self.lstm_dim, self.out_dim)), name='g_W_h_y')
        g_b_y = theano.shared(np.zeros((self.out_dim,)), name='g_B_y')

        return (g_W_xi, g_W_xo, g_W_xf, g_W_xc, g_W_hi, g_W_ho,
                g_W_hf, g_W_hc, g_W_ci, g_W_co, g_W_cf, g_W_hy,
                g_b_i, g_b_o, g_b_f, g_b_c, g_b_y)

    def init_model(self):
        """Return init_states, parameters, init_gradient_params
        the former is needed for momentum

        WARNING: Initial state is not being learned!
        """

        init_states = self._init_init_states()
        parameters = self._init_parameters()
        gradients = self._init_gradients()

        return init_states, parameters, gradients

    def _load_parameters(self, parameters_filename):
        with open(parameters_filename, "rb") as f:
            return pickle.load(f)

    def _load_model(self, parameters_filename):
        states = self._init_init_states()
        parameters = self._load_parameters(parameters_filename)
        gradients = self._init_gradients()

        return states, parameters, gradients

    def _step(self, x_t, c_tm1, h_tm1, W_xi, W_xo, W_xf, W_xc, W_hi, W_ho, W_hf,
             W_hc, W_ci, W_co, W_cf, W_hy, b_i, b_o, b_f, b_c, b_y):
        """LSTM step function
        from [Alex Graves, Generating Sequences With Recurrent Neural Networks]
        """
        # Input gate activation
        i_t = TT.nnet.sigmoid(TT.dot(x_t, W_xi) + TT.dot(h_tm1, W_hi) + c_tm1 * W_ci + b_i)
        # Forget gate activation
        f_t = TT.nnet.sigmoid(TT.dot(x_t, W_xf) + TT.dot(h_tm1, W_hf) + c_tm1 * W_cf + b_f)
        # Cell state
        c_t = f_t * c_tm1 + i_t * TT.tanh(TT.dot(x_t, W_xc) + TT.dot(h_tm1, W_hc) + b_c)
        # Output gate activation
        o_t = TT.nnet.sigmoid(TT.dot(x_t, W_xo) + TT.dot(h_tm1, W_ho) + c_t * W_co + b_o)
        # Hidden vector, cell output
        h_t = o_t * TT.tanh(c_t)
        # Output layer
        y_t = TT.dot(h_t, W_hy) + b_y
        return c_t, h_t, y_t

    def _loss_sse(self, y, t, m):
        """Sum of squared errors
        m is the mask to handle sequences with different lengths"""
        return TT.sum((((y - t) ** 2) * m))

    def _loss_mse(self, y, t, m):
        """Mean squared error
        m is the mask to handle sequences with different lengths"""
        return self._loss_sse(y, t, m) / m.sum()

    def _loss_rmse(self, y, t, m):
        """Root mean squared error
        m is the mask to handle sequences with different lengths"""
        return TT.sqrt(self._loss_mse(y, t, m))

    def _compute_gradients(self, loss, params):
        """Compute gradients of the parameters, given a loss variable"""
        grads = TT.grad(loss, params)
        return zip(params, grads)

    def _compute_update_rules(self, gradients, prev_updates, lr, momentum):
        """Compute param updates with momentum"""
        assert len(gradients) == len(prev_updates)
        updates = []
        for i in range(len(gradients)):
            w = gradients[i][0]  # Weight
            g = gradients[i][1]  # Weight gradient
            g0 = prev_updates[i]  # Previous weight update
            # First update g0 (previous weight update)
            updates.append((g0, momentum * g0 - lr * g))
            # Then update weights
            updates.append((w, w + g0))
        return updates

    def _shared_profile(self, shared_list):
        return reduce(
            lambda acc, x: acc + np.absolute(x.get_value()).sum(),
            shared_list, 0)

    def _num_total_weights(self, shared_list):
        return reduce(
            lambda acc, x: acc + x.get_value().size, shared_list, 0)

    def train(self):
        theano.config.profile = False
        np.set_printoptions(precision=3)
        theano.config.floatX = 'float32'
        theano.config.exception_verbosity = 'high'

        print 'Loading data...',
        ti = time.clock()
        Xtr, Ytr, Mtr = load_ncdata('corpora/anbncn_1_10.nc')
        Xte, Yte, Mte = load_ncdata('corpora/anbncn_10_100.nc')
        print time.clock() - ti
        print "Data shapes: Xtr={}, Ytr={}, Xte={}, Yte={}".format(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)

        # Inputs symbolic variable
        x = TT.tensor3(name='x')
        # Targets symbolic variable
        t = TT.tensor3(name='t')
        # Masks for controlling length
        ma = TT.tensor3(name='m', dtype='int8')
        # Learning rate, momentum
        lr = TT.scalar(name='lr')
        mo = TT.scalar(name='momentum')

        # Symbolic recursive expression
        print 'Creating recursive expression (scan)...',
        ti = time.clock()
        [_, _, y], _ = theano.scan(
            fn=self._step,
            sequences=[x],
            outputs_info=[TT.alloc(p, x.shape[1], self.lstm_dim) for p in self.init_state] + [None],
            non_sequences=self.lstm_params)
        print time.clock() - ti

        print 'Creating GD update rules...',
        ti = time.clock()
        # Symbolic loss variables
        loss_sse_v = self._loss_sse(y, t, ma)
        loss_mse_v = self._loss_mse(y, t, ma)
        loss_rmse_v = self._loss_rmse(y, t, ma)

        # Symbolic update rules
        updates = self._compute_update_rules(
            self._compute_gradients(loss_mse_v, self.lstm_params), self.lstm_prev_updates, lr, mo)
        print time.clock() - ti

        print 'Compiling train function...',
        ti = time.clock()
        train_fn = theano.function(
            [x, t, ma, lr, mo], [], updates=updates)
        print time.clock() - ti

        print 'Compiling eval function...',
        ti = time.clock()
        eval_fn = theano.function([x], y)
        eval_loss_fn = theano.function(
            [x, t, ma], [loss_sse_v, loss_mse_v, loss_rmse_v])
        print time.clock() - ti

        print 'Training...'

        # Print the number of weights to learn
        sys.stdout.write(
            'Trainable weights = %d\n' % self._num_total_weights(self.lstm_params))

        # Initial loss and parameters profile
        tr_sse, tr_mse, tr_rmse = eval_loss_fn(Xtr, Ytr, Mtr)
        te_sse, te_mse, te_rmse = eval_loss_fn(Xte, Yte, Mte)
        wpr = self._shared_profile(self.lstm_params)
        print ('EP={0:>2d}  TIME={1:>6.2f}  TR-MSE={2:f}  TE-MSE={3:f}  WPROF={4:>6.2f}  (t={5:>6.2f})'
                         .format(0, 0, tr_mse.item(), te_mse.item(), wpr, 0))

        # Start epochs
        p = [i for i in range(Xtr.shape[1])]  # Indexes for the train sequences
        batch_indexes = range(0, Xtr.shape[1], self.batch_size)  # Index of each batch
        SEQ_LEN, SEQ_DIM = Xtr.shape[0], Xtr.shape[2]
        tot_time = 0
        for ep in range(1, self.max_epochs + 1):
            ti = time.clock()
            # Mini-Batch gradient descent
            np.random.shuffle(p)   # Randomize data presentation
            for bi in batch_indexes:
                ep_bs = p[bi:bi + self.batch_size]  # batch sequences
                # Perform weights update
                train_fn(
                    Xtr[:, ep_bs, :].reshape((SEQ_LEN, len(ep_bs), SEQ_DIM)),
                    Ytr[:, ep_bs, :].reshape((SEQ_LEN, len(ep_bs), SEQ_DIM)),
                    Mtr[:, ep_bs, :].reshape((SEQ_LEN, len(ep_bs), SEQ_DIM)),
                    self.learning_rate, self.momentum)

            # Compute costs after all batches have updated weights
            tr_sse, tr_mse, tr_rmse = eval_loss_fn(Xtr, Ytr, Mtr)
            te_sse, te_mse, te_rmse = eval_loss_fn(Xte, Yte, Mte)
            wpr = self._shared_profile(self.lstm_params)  # Profile of new weights
            ep_ti = time.clock() - ti  # Epoch time
            tot_time += ep_ti  # Total elapsed time
            # Print status
            print ('EP={0:>2d}  TIME={1:>6.2f}  TR-MSE={2:f}  TE-MSE={3:f}  WPROF={4:>6.2f}  (t={5:>6.2f})'
                   .format(ep, tot_time, tr_mse.item(), te_mse.item(), wpr, ep_ti))
            if tot_time >= self.max_time:
                break

        # x_sample = Xtr[:, 0, :].reshape((SEQ_LEN, len(ep_bs), SEQ_DIM))
        # print x_sample
        # print eval_fn(x_sample)


if __name__ == '__main__':
    # from hinton import plot
    # Xtr, Ytr, Mtr = load_ncdata('corpora/anbncn_1_10.nc')
    # for i in xrange(2):
    #     plot(Xtr[:, i, :])
    #     plot(Ytr[:, i, :])
    #     print
    network = LSTM_RNN(seed=0)
    network.train()