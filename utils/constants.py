SEED = 1
MAX_FLOW_LEN = 100 # number of packets
TIME_WINDOW = 10
TRAIN_SIZE = 0.90 # size of the training set wrt the total number of samples

PATIENCE = 10
DEFAULT_EPOCHS = 1000
VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters','Validation Set']
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
HYPERPARAM_GRID = {
    "optimizer__learning_rate": [0.1, 0.01],  # ← 여기에 넘겨야 실제로 optimizer에 반영됨
    "batch_size": [1024,2048],
    "model__kernels": [32,64],
    "model__regularization" : [None,'l1'],
    "model__dropout" : [None,0.2]
}