import argparse
import numpy as np
import pickle
import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential, load_model
from keras.layers import Conv2D
import matplotlib.gridspec as gridspec
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import sklearn
from keras.regularizers import l2,l1
from keras.layers import *
from sklearn.metrics import hamming_loss,label_ranking_loss,confusion_matrix, auc,roc_curve,roc_auc_score, precision_recall_curve
from keras.optimizers import SGD
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras import optimizers,losses,metrics
from sklearn import metrics as met
import keras.backend as K
import keras.backend.tensorflow_backend as tfb
from skmultilearn.utils import measure_per_label
import sklearn.metrics as skm
import tensorflow as tf
# from tensorflow_addons.metrics import HammingLoss
import scipy
from numpy import asarray
from numpy import ones
from sklearn.metrics import fbeta_score
from sklearn import svm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
globalArr= [[],[],[],[],[],[],[],[]]
f_globalArr= [[],[],[],[],[],[],[],[]]
highest_roc = (-sys.maxsize - 1,0,0)
highest_prc = (-sys.maxsize - 1,0,0)
highest_accuracy = -sys.maxsize - 1
extension='single'

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        predValid = self.model.predict_proba(self.validation_data[0])
        pred_binary = K.round(predValid)
        Yvalid = self.validation_data[1]
        file = open("output-"+extension+".txt", "w")
        file.write('Epoch :'+str(batch))
        for i in range(50):
            temp = np.array_str(np.array([pred_binary[i,:],Yvalid[i,:].astype(int)]))
            file.write(temp)
            file.write('\n--------------\n')
        self.acc_per_label = measure_per_label(skm.accuracy_score, scipy.sparse.csr_matrix(Yvalid),scipy.sparse.csr_matrix(pred_binary))
        self.f_score = measure_per_label(my_fbeta, scipy.sparse.csr_matrix(Yvalid),scipy.sparse.csr_matrix(pred_binary))
        # self.prc_by_label = measure_per_label(skm.auc(skm.precision_recall_curve[0],skm.precision_recall_curve[1]), scipy.sparse.csr_matrix(Yvalid),scipy.sparse.csr_matrix(predValid.round()))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(Yvalid.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(Yvalid[:, i], pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        global roc_triplet
        roc_triplet = (roc_auc,fpr,tpr)
        self.auc_per_label = roc_auc

        prec = dict()
        recall = dict()
        prc_auc = dict()
        for i in range(Yvalid.shape[1]):
            prec[i], recall[i], _ = precision_recall_curve(Yvalid[:, i], pred_binary[:, i])
            prc_auc[i] = auc(recall[i], prec[i])
        global prc_triplet
        prc_triplet = (prc_auc,recall,prec)
        self.prc_per_label = prc_auc

        # confusion csr_matrix
        print("Confusion matrix: " ,skm.confusion_matrix(Yvalid, pred_binary))

        # print("Per label auc:",self.auc_per_label)
        # print("Per label prc:",self.prc_per_label)
        print("Per label accuracy:",self.acc_per_label)
        try:
            for i in range(8):
                globalArr[i].append(self.acc_per_label[i])
                f_globalArr[i].append(self.f_score[i])
        except:
            print("Single label scenario")
            f_globalArr[0].append(self.f_score) # my fbeta function is faulty
        print("Per label f2 score:",self.f_score)
        global highest_accuracy
        if logs.get('val_accuracy') > highest_accuracy:
            highest_accuracy = logs.get('val_accuracy')

        return



def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', np.unique(y_true[:, i]), y_true[:, i])
    return weights

def get_model(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length):
    model = Sequential()
    conv1_layer = Conv1D(filters=1000,
                        kernel_size=8,
                        input_shape=(length, 4),
                        padding="valid",
                        activation="relu",
                        # use_bias=True, kernel_regularizer=l2(0.001))
                        use_bias=True)
    model.add(conv1_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.2))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=4,
                        filters=500,
                        use_bias=True, kernel_regularizer=l2(0.001))
                        # use_bias=True)
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.2))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=4,
                        filters=250,
                        use_bias=True, kernel_regularizer=l2(0.001))
                        # use_bias=True)
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(units=numLabels, use_bias=True, kernel_regularizer=l2(0.001)))
    model.add(Activation('sigmoid'))
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    gs = gridspec.GridSpec(3, 2)

    fig = pyplot.figure()
    # plot loss
    ax1=pyplot.subplot(gs[0, :])
    ax1.title.set_text('Cross Entropy Loss')
    ax1.plot(history.history['loss'], color='blue', label='train')
    ax1.plot(history.history['val_loss'], color='orange', label='test')
    ax1.set_yticks(np.arange(0, 1.2, step=0.2))

	# # plot fbeta
    # ax2 = pyplot.subplot(411)
    # ax2.title.set_text('Fbeta')
    # if extension!='single':
    #     for i in range(8):
    #         ax2.plot(f_globalArr[i], label='mine')
    #     ax2.legend(
    #         ['Progenitor','Dendritic','Monocyte','B cell','Basophil','NK cell','CD4+','CD8+'],loc='lower left', ncol=2)
    # # else:
    # #     ax2.plot(f_globalArr[0], label='sklearn')
    # ax2.plot(history.history['fbeta'], color='blue', label='train')
    # ax2.plot(history.history['val_fbeta'], color='orange', label='test')
    # ax2.set_yticks(np.arange(0, 1.2, step=0.2))

    #plot accuracy
    ax3=pyplot.subplot(gs[1, :])
    ax3.title.set_text('Accuracy')
    if extension!='single':
        for i in range(8):
            ax3.plot(globalArr[i], label='mine')
        ax3.legend(['Progenitor','Dendritic','Monocyte','B cell','Basophil','NK cell','CD4+','CD8+'],loc='lower left', ncol=2)
    ax3.plot(history.history['accuracy'], label='train')
    ax3.plot(history.history['val_accuracy'], label='test')
    ax3.set_yticks(np.arange(0, 1.2, step=0.2))

    #Plot auroc
    ax4=pyplot.subplot(gs[2, 0])
    if extension!='single':
        for i in range(8):
            ax4.plot(roc_triplet[1][i],roc_triplet[2][i],label='ROC curve (area = %0.2f)' % roc_triplet[0][i])
    else:
        ax4.plot(roc_triplet[1][0],roc_triplet[2][0],label='ROC curve (area = %0.2f)' % roc_triplet[0][0])
    ax4.set_yticks(np.arange(0, 1.25, step=0.2))
    ax4.set_xticks(np.arange(0, 1.2, step=0.2))
    ax4.legend(loc="lower right")

    #Plot auprc
    ax5=pyplot.subplot(gs[2, 1])
    if extension!='single':
        for i in range(8):
            ax5.plot(prc_triplet[1][i],prc_triplet[2][i],label='PRC curve (area = %0.2f)' % prc_triplet[0][i])
    else:
        ax5.plot(prc_triplet[1][0],prc_triplet[2][0],label='PRC curve (area = %0.2f)' % prc_triplet[0][0])
    ax5.set_yticks(np.arange(0, 1.25, step=0.2))
    ax5.set_xticks(np.arange(0, 1.2, step=0.2))
    ax5.legend(loc="lower right")

    # save plot to file
    fig.tight_layout()
    fig.savefig('my_plot-new-' + extension+'.png')
    pyplot.close()

def train_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize,
                     numEpochs,
                     numConvLayers,
                     numConvFilters,
                     poolingDropout,
                     learningRate,
                     momentum,
                     length,
                     pretrainedModel):

    # class_weights = calculating_class_weights(Y_train)
    # print(class_weights)

    # numLabels=1
    try:
        numLabels = Y_train.shape[1]
    except:
        numLabels= 1
    # X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
    # X_valid = np.reshape(X_valid, (X_valid.shape[0],1,X_valid.shape[1],X_valid.shape[2]))
    if pretrainedModel:
        model = load_model(pretrainedModel)
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, poolingDropout, learningRate, momentum, length)
        optim = SGD(lr=learningRate, momentum=momentum)
    #'binary_crossentropy' get_weighted_loss(class_weights)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])#,fbeta]), ranking_loss]) #specificity_metric])
    model.summary()
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    earlystopper = EarlyStopping(patience=10, monitor='val_accuracy', min_delta=0, verbose=0, mode='max')
    print(X_valid.shape)
    print(Y_valid.shape)
    cust_metrics = Metrics()
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1,
    validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer,cust_metrics, earlystopper])#, class_weight = classWeights)
    # from skmultilearn.problem_transform import BinaryRelevance
    # from skmultilearn.ext import Keras
    # KERAS_PARAMS = dict(batch_size=batchSize, epochs=numEpochs, shuffle=True, verbose=1,
    # validation_data = (X_valid, Y_valid), initial_epoch=0, callbacks=[checkpointer,cust_metrics])
    # clf = BinaryRelevance(classifier=Keras(model, False, KERAS_PARAMS), require_dense=[False,True])
    # history = clf.fit(X_train, Y_train)
    # learning curves
    summarize_diagnostics(history)

def my_fbeta(y_true, y_pred):
    return fbeta_score(y_true, y_pred,2)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a convolutional neural networnp model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data', required=True)
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels', required=True)
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data', required=True)
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels', required=True)
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output', required=True)
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=50)
    parser.add_argument('-n', '--num-conv-layers', type=int, help='number of convolutional layers to use', required=False, default=2)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use in layers after the first one', required=False, default=100)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.9)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=499)
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model', required=False, default=None)
    parser.add_argument('-c1w', '--class-1-weight', type=int, help='weight for positive class during training', required=False, default=1)
    parser.add_argument('-c2w', '--class-2-weight', type=int, help='weight for positive class during training', required=False, default=1)
    args = parser.parse_args()
    print("Loading data")
    X_train = np.load(file=args.xtrain)
    Y_train = np.load(file=args.ytrain)

    X_valid = np.load(file=args.xvalid)
    Y_valid = np.load(file=args.yvalid)

    # Note no. of samples will be doubled from output of standardizepeaks
    # because of reverse complements
    print("Training on: "+ str(X_train.shape[0]))
    print("Validating on: " + str(X_valid.shape[0]))

    # if extension=='single':
    #     print("Extracting 1 label")
    #     Y_train = Y_train[:,1]
    #     Y_valid = Y_valid[:,1]

    #
    print('Model state:')
    print('LR='+str(args.learning_rate)+'\nMomentum='+str(args.momentum))
    #
    #train SVM models
    #use gkm-svm : lsgkm
    # ''' To generate fasta files use:
    # sed '/^chr8/ d' combined.bed | sed '/^chr9/ d' | sed '/^chr4/ d' > training.bed
    # ./bedtools getfasta -fi hg38.fa -bed diff_acess/negSet-2.bed -fo diff_acess/negSet.fa
    # ./bedtools getfasta -fi hg38.fa -bed diff_acess/training.bed -fo diff_acess/posSet.fa '''
    # print("Training SVM model")
    # os.system('/home/snigdhaa/lsgkm/src/gkmtrain -T 16 -s /home/snigdhaa/diff_acess/svmData/prom2pos.fa /home/snigdhaa/diff_acess/svmData/prom2neg.fa svmModel-short')

    print(Y_valid.shape)
    test_yhat = asarray([np.zeros(Y_valid.shape[1]) for _ in range(Y_valid.shape[0])])
    print(measure_per_label(skm.accuracy_score, scipy.sparse.csr_matrix(Y_valid),scipy.sparse.csr_matrix(test_yhat)))
    m = tf.keras.metrics.Accuracy()
    _ = m.update_state(Y_valid,test_yhat)
    print(m.result().numpy() )

    #Check baseline auroc and auprc for single label
    prec, recall, _ = precision_recall_curve(Y_valid, test_yhat)
    prc_auc = auc(recall, prec)
    print("Baseline prc:"+str(prc_auc))

    fpr, tpr, _ = roc_curve(Y_valid, test_yhat)
    roc_auc = auc(fpr, tpr)
    print("Baseline roc:"+str(roc_auc))

    train_model(modelOut=args.model_out,
                     X_train=X_train,
                     Y_train=Y_train,
                     X_valid=X_valid,
                     Y_valid=Y_valid,
                     batchSize=args.batch_size,
                     numEpochs=args.num_epochs,
                     numConvLayers=args.num_conv_layers,
                     numConvFilters=args.num_conv_filters,
                     poolingDropout=args.pool_dropout_rate,
                     learningRate=args.learning_rate,
                     momentum=args.momentum,
                     length=args.length,
                     pretrainedModel=args.pretrained_model)
    print('Highest accuracy: '+str(highest_accuracy))

    model = load_model('output-newDrop.hdf5')
    predValid = model.predict_proba(X_valid)
    pred_binary = K.round(predValid)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(Y_valid.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], pred_binary[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_triplet = (roc_auc,fpr,tpr)

    prec = dict()
    recall = dict()
    prc_auc = dict()
    for i in range(Y_valid.shape[1]):
        prec[i], recall[i], _ = precision_recall_curve(Y_valid[:, i], pred_binary[:, i])
        prc_auc[i] = auc(recall[i], prec[i])
    prc_triplet = (prc_auc,recall,prec)

    gs = gridspec.GridSpec(1, 2)

    fig = pyplot.figure()
    #Plot auroc
    ax4=pyplot.subplot(gs[0, 0])
    if extension!='single':
        for i in range(8):
            ax4.plot(roc_triplet[1][i],roc_triplet[2][i],label='ROC curve (area = %0.2f)' % roc_triplet[0][i])
    else:
        ax4.plot(roc_triplet[1][0],roc_triplet[2][0],label='ROC curve (area = %0.2f)' % roc_triplet[0][0])
    ax4.set_yticks(np.arange(0, 1.25, step=0.2))
    ax4.set_xticks(np.arange(0, 1.2, step=0.2))
    ax4.legend(loc="lower right")

    #Plot auprc
    ax5=pyplot.subplot(gs[0, 1])
    if extension!='single':
        for i in range(8):
            ax5.plot(prc_triplet[1][i],prc_triplet[2][i],label='PRC curve (area = %0.2f)' % prc_triplet[0][i])
    else:
        ax5.plot(prc_triplet[1][0],prc_triplet[2][0],label='PRC curve (area = %0.2f)' % prc_triplet[0][0])
    ax5.set_yticks(np.arange(0, 1.25, step=0.2))
    ax5.set_xticks(np.arange(0, 1.2, step=0.2))
    ax5.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig('auc_plot-new-' + extension+'.png')
    pyplot.close()
