def ROC_curve(data, labels, model, name, color):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    if color == 'red':
        t = 'Train'
    else:
        t = 'Test'

    # use the model to do classifications
    #AI: this retutns numpy array with shape (nEntries,1) :
    labels_predict = model.predict(data)

    #AI: fpr = false positive rate; tpr = true positive rate
    fpr, tpr, thresholds = roc_curve(
        labels, labels_predict[:, 0])  # calculate the ROC curve
# Disable weights
#    fpr, tpr, thresholds = roc_curve(
#        labels, labels_predict[:, 0])  # calculate the ROC curve        
    #AI: compute the area under the roc curve
    roc_auc = auc(fpr, tpr)
    print("fpr = ", fpr)
    print("tpr = ",tpr)
    print("thresholds = ",len(thresholds), "  ", thresholds)
    print ("roc_auc = ", roc_auc)
    if color == 'red':
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='k', label='random chance')
    plt.plot(tpr, fpr, lw=2, color=color, label=t+' NN auc = %.3f' % (roc_auc))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('true positive rate')
    plt.ylabel('false positive rate')
    plt.title('receiver operating curve')
    plt.legend(loc="upper left")
    plt.grid()
    if 'testing' in name:
        plt.savefig('plots/{}.png'.format(name))


def trainingPlots(history, name):
    import matplotlib.pyplot as plt
    # plot loss vs epoch
    ax = plt.subplot(2, 1, 1)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.legend(loc="upper right")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    # plot accuracy vs epoch
    ax = plt.subplot(2, 1, 2)
    ax.plot(history.history['accuracy'], label='acc')
    ax.plot(history.history['val_accuracy'], label='val_acc')
    ax.legend(loc="upper left")
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    plt.savefig('plots/{}.png'.format(name))


def discPlot(name, model, train_sig, train_bkg, test_sig, test_bkg):
    import matplotlib.pyplot as plt
    import numpy as np

    train_sig_pred = model.predict(train_sig)
    train_bkg_pred = model.predict(train_bkg)
    test_sig_pred = model.predict(test_sig)
    test_bkg_pred = model.predict(test_bkg)

    plt.figure(figsize=(12, 8))
    plt.title('NN Discriminant')
    plt.xlabel('NN Disc.')
    plt.ylabel('Events/Bin')

    nb, binb, _ = plt.hist(test_bkg_pred, bins=50, range=(0, 1), density=True)
    bin_centers = 0.5*(binb[1:] + binb[:-1])
    ns, bins, _ = plt.hist(test_sig_pred, bins=50, range=(0, 1), density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])

    plt.clf()

    plt.hist(train_bkg_pred, histtype='stepfilled', color='red', label='ttbar Train', bins=50, range=(0,1), density=True, alpha=0.5)
    plt.hist(train_sig_pred, histtype='stepfilled', color='blue', label='HH Train', bins=50, range=(0,1), density=True, alpha=0.5)


    plt.errorbar(y=nb, x=bin_centers, yerr=np.sqrt(nb)*.1, fmt='o', color='pink', label='ttbar Test')
    plt.errorbar(y=ns, x=bin_centers, yerr=np.sqrt(ns)*.1, fmt='o', color='cyan', label='HH Test')

    plt.legend()
    plt.savefig('plots/{}.png'.format(name))
