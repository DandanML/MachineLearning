# This file gives an example on how to process file stored on S3 directly.
# also, gives an example on how to concatenate the output of CNN/MaxPooling with additional input, and then use the concatenated results as input to Dense(fully connected) layers.

import numpy as np
import time
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.layers import Conv2D, MaxPooling2D,Flatten, concatenate
from keras.callbacks import CSVLogger
import pandas as pd
import gzip
import boto3
import io

#using keras with tensorflow backend
def cnn_model(input_shape_cnn, additional_input_shape, output_dim, numofnodes, f1, f2,k1, k2, k3, k4, p1, p2, s1, s2):
    # this model is concatenate the output of CNN with additional input and then use them as input to FC DNN layer.    
   cnn_inputs = Input(shape=input_shape_cnn, name='cnn_input')
    x = Conv2D(f1, kernel_size=(k1, k2), padding='same', strides=(s1, s1),
                     activation='relu',input_shape=input_shape_cnn) (cnn_inputs)
    x = MaxPooling2D(pool_size=(p1, p1), strides=(p1, p1))(x)
    x = Conv2D(f2, (k3, k4), padding='same', activation='relu', strides=(s2,s2))(x)
    x = MaxPooling2D(pool_size=(p2, p2), strides=(p2, p2))(x)
    cnn_output = Flatten()(x)
    dnn_input = Input(shape=additional_input_shape, name='dnn_input')
    x = concatenate([cnn_output, dnn_input])
    x = Dense(numofnodes, activation = 'relu')(x)
    predictions = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=[cnn_inputs, dnn_input], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train_cnn_input, X_train_dnn_input, dummy_y_train, input_shape_cnn, additional_input_shape, output_dim,numofnodes, f1, f2, k1, k2, k3, k4, p1, p2, s1, s2):
    model = cnn_model(input_shape_cnn, additional_input_shape, output_dim,numofnodes, f1, f2, k1, k2, k3, k4, p1, p2, s1, s2)
    num_parameter = model.count_params()
    print(model.summary())
    tic = time.clock()
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, verbose=2, mode='auto') 
	log_filename ='training_result.csv'
    model_filename = "cnn_model_" + ".json"
    model_weight_filename = "cnn_model_"+ ".h5"
    encoder_class_filename= "encoder_class.npy"
	csv_logger = CSVLogger(log_filename, append=False, separator=',')
    checkpoint = ModelCheckpoint(model_weight_filename, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [early_stop, csv_logger, checkpoint]
	#note that earlyStopping is good for initial fast prototyping. After choosing the model, it is better to let it train longer time to get the accuracy. 
    model.fit([X_train_cnn_input, X_train_dnn_input], dummy_y_train, epochs=200, batch_size=512, validation_split=0.1, validation_data=None,
              callbacks=callbacks_list, verbose=2, shuffle=True)
    toc = time.clock()
    # save the model
 #   model_json = model.to_json()
#    with open(model_filename, "w") as json_file:
#        json_file.write(model_json)
#    model.save_weights(model_weight_filename)
#    print("Saved model to disk")

    print("Took %d seconds to train the model" % (toc - tic))
    print("\nModel Evaluation on training data")
    score = model.evaluate(X_train, dummy_y_train, verbose=2)
    print(score[1])
    return num_parameter, score[1]

if __name__ == '__main__':
	data_filename = "xxx.csv.gz"
    bucket = 'temp-datasets' #bucketname
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=data_filename)
    image_row_num = 8
    image_column_num = 8
    tic = time.clock()
    print("\nReading dataset from file")
    input_data_frame = pd.read_csv(io.BytesIO(obj['Body'].read()), header=None, compression='gzip')
    values = input_data_frame.values
    num_samples = values.shape[0]
    num_features = values.shape[1] - 2  # subtract 2 to get rid of first column which are row numbers, and last column which is the label.
    X_train = values[:, 1:num_features + 1] # the dimension of each sample is 1x(image_row_num*image_column_num+2*image_row_num), 2 is the additional feature for each row.
    X_train = X_train.astype('float32')
    y_train = values[:, num_features + 1]
    num_features_cnn = image_row_num * image_column_num
    X_train_cnn_input = X_train[:,0:num_features_cnn]
    X_train_additional_input = X_train[:,num_features_cnn:]
    print(X_train_additional_input.shape)
    toc = time.clock()
    print("Took %d seconds to read the data file"%(toc-tic))
    encoder = LabelEncoder()
    encoder.fit(y_train)
   # np.save(encoder_class_filename, encoder.classes_)
    encoded_y_train = encoder.transform(y_train) # convert output categories into numbers based
    dummy_y_train = np_utils.to_categorical(encoded_y_train) #one hot vector
    output_dim = dummy_y_train.shape[1]
    X_train_cnn_input = X_train_cnn_input.reshape(num_samples, image_row_num, image_column_num, 1)
    input_shape_cnn = (image_row_num, image_column_num,1)
    additional_input_shape = (2*image_row_num,) # 2*image_row_num are the additional input features.
    f1_array = [32]
    f2_array = [32]
    k1_array = [2]
    k2_array = [2]
    k3_array = [2]
    k4_array = [2]
    p1_array = [2]
    p2_array = [2]
    s1_array = [1]
    s2_array = [1]
	numofnodes = 32
    for f1 in f1_array:
        for f2 in f2_array:
            for k1 in k1_array:
                for k2 in k2_array:
                    for k3 in k3_array:
                        for k4 in k4_array:
                            for p1 in p1_array:
                                for p2 in p2_array:
                                    for s1 in s1_array:
                                        for s2 in s2_array:
                                            if (4/s1/p1/s2/p2>=1):
                                                [total_parameter, accuracy] = train_model(X_train_cnn_input, X_train_additional_input, dummy_y_train, input_shape_cnn, additional_input_shape, output_dim,numofnodes, f1, f2, k1, k2, k3, k4,p1,p2,s1, s2)
                                                result = "f1="+str(f1)+" f2="+str(f2)+ " k1="+str(k1)+" k2="+str(k2)+" k3="+str(k3)\
                                                         +" k4="+str(k4)+" p1="+str(p1)+" p2="+str(p2)+" s1="+str(s1)+" s2"+str(s2)\
                                                         +" totalpara="+str(total_parameter)+" accuracy="+str(accuracy)
                                                #k2, k3, k4,p1,p2,s1, s2, total_parameter, accuracy])
                                                print(result)
                                                hp = open("myresults.txt", "a")
                                                hp.write(result+"\n")
                                                hp.close()
