import tensorflow as tf
import sklearn as sk
from sklearn.cross_decomposition import CCA
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()


class CCA_numpy:

    def __init__(self, xMatrix, yMatrix):
        self.x = xMatrix
        self.y = yMatrix
        print("initialized fine")

    def changeToNumpy(self):

        tf.compat.v1.enable_eager_execution()
        print('in changeToNumpy')
        x = self.x.numpy()
        y = self.y.numpy()
        print('past changeToNumpy')



        return x, y


    def changeToTensor(self, x, y):

        x = tf.convert_to_tensor(value=x)
        y = tf.convert_to_tensor(value=y)

    def doCCA(self):

        x, y = self.changeToNumpy()

        cca = CCA(n_components=1)
        cca.fit(x, y)
        print(cca.coef_.shape)

        x_c, y_c = cca.transform(x, y)
        result = np.corrcoef(x_c.T, y_c.T).diagonal(offset=1)

        print(result)


        return result





class CCA_cdp(object):

    def __init__(self, xMatrix, yMatrix):

        print('initialized CCA')

        self.x = xMatrix
        self.y = yMatrix
        print('initialized CCA')


    def getCovarianceMatrix(self, matrix):
        print('in get Covariance')
        tf.print(matrix.shape)
        transMatrix = tf.transpose(a=matrix)

        centredMatrix = transMatrix - tf.matmul(tf.eye(matrix.shape[1]), tf.expand_dims(tf.reduce_mean(input_tensor=matrix, axis=0),axis=-1))

        SampleCov = tf.math.scalar_mul(1/(matrix.shape[1]),tf.matmul(tf.transpose(a=centredMatrix), centredMatrix))

        print('got covariance')

        return SampleCov


    def getCrossCovMatrix(self, a, b):

        print('in cross cov')

        meanA = tf.reduce_mean(input_tensor=a, axis=1)
        meanB = tf.reduce_mean(input_tensor=b, axis=1)

        centredA = tf.transpose(a=a) - tf.matmul(tf.eye(a.shape[1]), meanA)
        centredB = tf.transpose(a=b) - tf.matmul(tf.eye(b.shape[1]), meanB)

        SampleCrossCov = tf.math.scalar_mul(2/(a.shape[1] +b.shape[1]),tf.matmul(tf.transpose(a=centredA), centredB))

        print('got cross cov')

        return SampleCrossCov



    def CCA(self):
        print('in cca')

        xCov = self.getCovarianceMatrix(self.x)
        yCov = self.getCovarianceMatrix(self.y)
        cross = self.getCrossCovMatrix(self.x, self.y)

        invHalf11 = tf.linalg.inv(self.symsqrt(xCov))

        inv22 = tf.linalg.inv(yCov)


        a = tf.matmul(invHalf11, cross)

        b = tf.matmul(tf.transpose(a=cross), invHalf11)

        c = tf.matmul(a, inv22)

        tilde = (c, b)

        eigens, vectors = tf.linalg.eigh(tilde)

        totalCorr = tf.math.reduce_sum(input_tensor=eigens)

        print('got cca')

        return totalCorr


    def symsqrt(mat, eps=1e-7):
        """Symmetric square root."""
        s, u, v = tf.linalg.svd(mat)
        # sqrt is unstable around 0, just use 0 in such case
        si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
        return u @ tf.linalg.tensor_diag(si) @ tf.transpose(a=v)







