import argparse
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras import optimizers
from keras.regularizers import l2,l1
import keras.backend as K
from numpy import asarray

import keras.metrics
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['svg.fonttype'] = 'none'
# rcParams['font.size']=10
import matplotlib.ticker as ticker
from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
import pandas as pd
import xlsxwriter
from itertools import *
import CLR
from CLR.clr_callback import CyclicLR
from CLR import config
from scipy.stats import mannwhitneyu
import rpy2
from rpy2.robjects.packages import importr
graphics = importr('graphics')
from rpy2.robjects import numpy2ri
numpy2ri.activate()
grdevices = importr('grDevices')
base = importr('base')

path = ""

def pearson_correlation(x, y):
    a_mean = K.mean(x)
    b_mean = K.mean(y)
    a_norm = x - a_mean
    b_norm = y - b_mean
    numerator = K.sum(a_norm * b_norm)

    a_var = K.sum(K.square(a_norm))
    b_var = K.sum(K.square(b_norm))
    denominator = (a_var * b_var) ** 0.5

    r= numerator / denominator
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

pearson_list = []
spearman_list = []
class Metrics(Callback):
    def __init__(self,train_data):
        self.training_data = train_data

    def on_epoch_end(self, batch, logs={}):
        predValid = self.model.predict(self.validation_data[0])
        Yvalid = self.validation_data[1]
        pearson,pval = pearsonr(Yvalid.flatten(),predValid.flatten())
        pearson_list.append(pearson)
        print("Validation Pearson: ",pearson)
        spearman,pval = spearmanr(Yvalid.flatten(),predValid.flatten())
        spearman_list.append(spearman)
        print("Validation Spearman: ",spearman)
        # predicted vs actual
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax2= axs[0]
        ax2.title.set_text('Validation')
        ax2.plot(Y_valid, predValid,'r.',alpha=0.5)
        ax2.set_aspect('equal')
        #plot training data actual vs predicted
        pred_train = self.model.predict(self.training_data[0])
        Y_train = self.training_data[1]
        ax3=axs[1]
        ax3.title.set_text('Training')
        # ax3.imshow(graphics.smoothScatter(Y_train,pred_train))
        ax3.plot(Y_train, pred_train,'r.',alpha=0.5)
        ax3.set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path+'actualvspred.png')
        pyplot.close()


def summarize_diagnostics(history):
    gs = gridspec.GridSpec(2,1)
    fig = pyplot.figure()

    ax2=pyplot.subplot(gs[0, :])
    ax2.title.set_text('Loss')
    ax2.plot(history.history['val_loss'], color='blue', label='test')
    ax2.plot(history.history['loss'], color='orange', label='train')
    ax2.legend(loc="lower right")

    ax3=pyplot.subplot(gs[1, :])
    ax3.title.set_text('Pearson')
    colors = ['green','pink','red','cyan','magenta']
    ax3.plot(history.history['val_pearson_correlation'], color='blue', label='test')
    ax3.plot(history.history['pearson_correlation'], color='orange', label='train')
    ax3.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path+'loss_pearson.png')
    pyplot.close()

def get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length):
    model = Sequential()
    l1_reg = 0.001
    l2_reg = 0.01
    l3_reg = 0.01
    l4_reg = 0.01
    dropout = 0.25
    filter1 = 500
    filter2 = 250
    filter3 = 100
    conv1_layer = Conv1D(filters=filter1,
                        kernel_size=8,
                        input_shape=(length, 4),
                        padding="valid",
                        activation="relu",
                        use_bias=True, kernel_regularizer=l2(l1_reg))
    model.add(conv1_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(dropout))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=8,
                        filters=filter2,
                        use_bias=True, kernel_regularizer=l2(l2_reg))
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(dropout))

    convn_layer = Conv1D(padding="valid",
                        activation="relu",
                        kernel_size=8,
                        filters=filter3,
                        use_bias=True, kernel_regularizer=l2(l3_reg))
    model.add(convn_layer)
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(units=numLabels, activation='relu', use_bias=True, kernel_regularizer=l2(l4_reg)))
    print("Regularization values: ",l1_reg,l2_reg,l3_reg,l4_reg)
    print("Dropout values: ",dropout)
    print("Filter values: ", filter1,filter2,filter3)
    return model

def weighted_mse(yTrue,yPred):
    # higher weight for negative samples
    # bool_arr = K.cast(K.less(yTrue, 3),dtype='float32')
    # higher weight for greater than 0.75 signal value
    bool_arr = K.cast(K.greater(yTrue, 4),dtype='float32')
    m = K.ones_like(bool_arr)*2
    w2 = bool_arr * m
    # w = K.log(yTrue + 1) + 1
    # return K.mean(K.square(yTrue-yPred)) * (w2+1) * w
    return K.mean(K.square(yTrue-yPred))*(w2+1)

def custom_loss(yTrue,yPred):
    return 1.0-pearson_correlation(yTrue,yPred)

def train_model(modelOut,
                     X_train,
                     Y_train,
                     X_valid,
                     Y_valid,
                     batchSize,
                     numEpochs,
                     numConvLayers,
                     numConvFilters,
                     preLastLayerUnits,
                     poolingDropout,
                     learningRate,
                     momentum,
                     length,
                     pretrainedModel,
                    ):
    numLabels = 1
    optim = optimizers.SGD(lr=learningRate, momentum=momentum)
    # print("Using cyclic LR")
    clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size= config.STEP_SIZE * (X_train.shape[0] // batchSize))
    if pretrainedModel:
        model = load_model(pretrainedModel, custom_objects={'weighted_mse': weighted_mse,'pearson_correlation':pearson_correlation})
        model.compile(loss='mean_squared_error', optimizer=optim, metrics=[pearson_correlation,'mse'])
    else:
        model = get_model(numLabels, numConvLayers, numConvFilters, preLastLayerUnits, poolingDropout, learningRate, momentum, length)
        l = weighted_mse#'mean_squared_error' #
        print("Using loss: ",l)
        model.compile(loss=l, optimizer=optim, metrics=[pearson_correlation,'mse'])

    model.summary()
    cust_metrics = Metrics((X_train,Y_train))
    checkpointer = ModelCheckpoint(filepath=modelOut,
                                   verbose=1, save_best_only=True, monitor='val_pearson_correlation', mode='max')
    earlystopper = EarlyStopping(monitor='val_pearson_correlation', min_delta=0, patience=15, verbose=0, mode='max')
    history = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=numEpochs,
    shuffle=True, verbose=1, validation_data = (X_valid, Y_valid), initial_epoch=0,
    callbacks=[cust_metrics, checkpointer,  earlystopper,clr])

    # generate actual vs pred plot for best saved model
    model = load_model(modelOut, custom_objects={'pearson_correlation':pearson_correlation,'weighted_mse': weighted_mse})
    pred_valid = model.predict(X_valid)

    # fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    fig, ax3 = plt.subplots(figsize=(5,5))
    for axis in [ax3.xaxis, ax3.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    pearson,pval = pearsonr(Y_valid.flatten(),pred_valid.flatten())
    # ax2 = axs[0]
    ax3.plot(Y_valid, pred_valid,'r.',alpha=0.5)
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    lims = [
    np.min([ax3.get_xlim(), ax3.get_ylim()]),  # min of both axes
    np.max([ax3.get_xlim(), ax3.get_ylim()]),  # max of both axes
    ]
    ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax3.text(1, 0, r'PR = {0:.2f}'.format(pearson),ha='right', va='bottom',
    transform=ax3.transAxes,fontsize=12)

    # ax3 = axs[1]
    # ax3.title.set_text('mono')
    # heatmap, xedges, yedges = np.histogram2d(Y_valid,
    #                                          pred_valid.flatten(),
    #                                          bins=100
    #                                         )
    #
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #
    # ax3.imshow(heatmap.T,
    #            extent=extent,
    #            origin='lower',
    #            cmap='Reds'
    #           )
    # ax3.imshow(graphics.smoothScatter(Y_valid,pred_valid.flatten()))
    # ax3.set_xlabel("Actual")
    # ax3.set_ylabel("Predicted")
    # ax3.text(1, 0, r'PR = {0:.2f}'.format(pearson),ha='right', va='bottom',
    # transform=ax3.transAxes,fontsize=12)

    # ax3.set_aspect('equal')
    # fig.tight_layout()
    fig.savefig(path+'actualvspred_end.svg')

    grdevices.png(path+"actualvspred_smooth.png")
    graphics.par(cex=1.5)
    graphics.smoothScatter(Y_valid,pred_valid.flatten(),xlab="Actual",ylab="Predicted")
    graphics.legend(x='bottomright', legend=base.paste('PR =',base.round(pearson,2)))
    grdevices.dev_off()

    summarize_diagnostics(history)

def reverse_one_hot(sequence):
    chars = np.array(['A','G','C','T'])
    indices = np.argmax(sequence,axis=1)
    return ''.join(np.take(chars,indices))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a convolutional neural network model', fromfile_prefix_chars='@')
    parser.add_argument('-xt', '--xtrain', help='npy file containing training data')
    parser.add_argument('-yt', '--ytrain', help='npy file containing training labels')
    parser.add_argument('-xv', '--xvalid', help='npy file containing validation data')
    parser.add_argument('-yv', '--yvalid', help='npy file containing validation labels')
    parser.add_argument('-xT', '--xtest', help='npy file containing testing data')
    parser.add_argument('-o', '--model-out', help='hdf5 file path for output')
    parser.add_argument('-w', '--pretrained-model', help='path to hdf5 file containing pretrained model')
    parser.add_argument('-b', '--batch-size', type=int, help='mini-batch size for training', required=False, default=100)
    parser.add_argument('-e', '--num-epochs', type=int, help='number of epochs to train', required=False, default=200)
    parser.add_argument('-n', '--num-conv-layers', type=int, help='number of convolutional layers to use', required=False, default=4)
    parser.add_argument('-c', '--num-conv-filters', type=int, help='number of convolutional filters to use in layers after the first one', required=False, default=100)
    parser.add_argument('-u', '--pre-last-layer-units', type=int, help='number of sigmoid units in the layer before the output layer', required=False, default=100)
    parser.add_argument('-pdrop', '--pool-dropout-rate', type=float, help='dropout rate for pooling layer', required=False, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate for sgd optimizer', required=False, default=0.005)
    parser.add_argument('-m', '--momentum', type=float, help='momentum for sgd', required=False, default=0.9)
    parser.add_argument('-l', '--length', type=int, help='length of input nucleotide sequences', required=False, default=501)
    parser.add_argument('-md', '--mode', help='train or test ?', required=True)
    parser.add_argument('-d', '--directory', help='path to store all other files', required=True)

    args = parser.parse_args()
    path = args.directory
    mode = args.mode
    if mode=="train" :
        if (not args.xtrain or not args.ytrain or not args.xvalid or not args.yvalid or not args.model_out):
            print("Argument missing. Check xtrain, ytrain, xvalid, yvalid and model-out")
        else:
            X_train = np.load(file=args.xtrain)
            Y_train = np.load(file=args.ytrain)
            X_valid = np.load(file=args.xvalid)
            Y_valid = np.load(file=args.yvalid)

            #adding noise to Y_train and Y_valid for 0.0 labelled negative samples
            noise = np.random.normal(0.5, .1, Y_train.shape)
            y = Y_train < 1
            Y_train = y.astype(int) * noise + Y_train

            noise = np.random.normal(0.5, .1, Y_valid.shape)
            y = Y_valid < 1
            Y_valid = y.astype(int) * noise + Y_valid


            args.length = len(X_train[0])
            train_model(modelOut=args.model_out,
                         X_train=X_train,
                         Y_train=Y_train,
                         X_valid=X_valid,
                         Y_valid=Y_valid,
                         batchSize=args.batch_size,
                         numEpochs=args.num_epochs,
                         numConvLayers=args.num_conv_layers,
                         numConvFilters=args.num_conv_filters,
                         preLastLayerUnits=args.pre_last_layer_units,
                         poolingDropout=args.pool_dropout_rate,
                         learningRate=args.learning_rate,
                         momentum=args.momentum,
                         length=args.length,
                         pretrainedModel=args.pretrained_model
                        )
    elif mode=="test":
        if (not args.pretrained_model or not args.xtest or not args.yvalid):
            print("Argument missing. Check pretrained_model or xtest or yvalid")
        else:
            model = load_model(args.pretrained_model, custom_objects={'pearson_correlation':pearson_correlation,'weighted_mse': weighted_mse,})
            print("Predicting for ", args.xtest)
            X_test = np.load(file=args.xtest)
            print(X_test.shape)
            predValid = model.predict(X_test)

            # sequences=[]
            # for i in range(X_test.shape[0]):
            #     sequences.append(reverse_one_hot(X_test[i,:,:]))
            #
            # X_seq = np.stack(sequences, axis=0)
            # combined = np.hstack((X_seq.reshape(-1,1),predValid))
            # print(combined.shape)
            # # # Generate dataframe from list and write to xlsx.
            # xlsx_name= args.xtest.rsplit('/',1)[1].split(".npy")[0]
            # print("Saving in ", path+"/"+xlsx_name+".xlsx")
            # pd.DataFrame(combined).to_excel(str(path+"/"+xlsx_name+".xlsx"), engine='openpyxl',header=False, index=False)

            Y_valid = np.load(file=args.yvalid)
            fig, ax3 = plt.subplots(figsize=(3,3))
            for axis in [ax3.xaxis, ax3.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))

            ## Actual vs predicted plots for validationSet
            pearson,pval = pearsonr(Y_valid.flatten(),predValid.flatten())
            ## Smoothscatter
            grdevices.svg(path+"actualvspred_smooth.svg")
            graphics.par(cex=1.5)
            graphics.smoothScatter(Y_valid,predValid.flatten(),xlab="Actual",ylab="Predicted")
            graphics.legend(x='bottomright', legend=base.paste('PR =',base.round(pearson,2)))
            grdevices.dev_off()
            ## Simple plot
            # ax3.plot(Y_valid, predValid,'.',alpha=0.5,color='red',label='mono Peaks')
            # ax3.set_xlabel("Actual")
            # ax3.set_ylabel("Predicted")
            # ax3.text(1, 0, r'PR = {0:.2f}'.format(pearson),ha='right', va='bottom',
            # transform=ax3.transAxes,fontsize=12)

            ## Histogram plots - Monocyte predictions for mono and cd4 peaks
            # plt.hist(predValid,density=True,bins=50,color='red',label='mono', alpha=0.5)
            # ax3.set_xlabel("Predicted signal")
            # ax3.set_ylabel("Density")
            plt.xticks(rotation=90)
            #comparing against another peak set
            # valid_peaks = np.load(file='Archr/cd4/Cluster7_validationInput.npy')
            # predictions = model.predict(valid_peaks)
            # actual_values= np.load(file='Archr/cd4/Cluster7_validationLabels.npy')
            # # ax3.plot(actual_values, predictions,'.',alpha=0.5,color='blue',label='cd4 Peaks')
            # _,p = mannwhitneyu(predValid,predictions,alternative='two-sided')
            # # plt.hist(predictions,density=True,bins=50,color='blue',label='cd4', alpha=0.5)
            # if p==0:
            #     ax3.text(4.5,0.55,r'pValue<0.0005')
            # else:
            #     ax3.text(4.5,0.55,r'pValue = {0:.4f}'.format(p))
            # ax3.legend(loc="upper right")


            fig.tight_layout()
            fig.savefig(path+"/"+'actualvspred_end.svg')
