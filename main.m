
function []=main(epochNum)

% default value for parameter(s)
arguments
    epochNum (1,1) {} = 400
end

%% device settings
gpuDevice([]);  
gpu = gpuDevice(); 
disp(['Using GPU: ', gpu.Name]);

%% data loading adjustments
trainData = "./data/Train";
testData = "./data/Test";

imgStoreTrain = imageDatastore(trainData, "IncludeSubfolders", true, "LabelSource", "foldernames");
imgStoreTest = imageDatastore(testData, "IncludeSubfolders", true, "LabelSource","foldernames");

classNum = numel(categories(imgStoreTrain.Labels));


%% dataset random image visualizations

visualizeDataset(imgStoreTrain);


%% data preprocessing

inputSize = [224 224];

[imgStoreTrainSplit, imgStoreVal] = splitEachLabel(imgStoreTrain, 0.8, 'randomized');


imgAugment = imageDataAugmenter( ...
    'RandRotation', [-10,10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize, imgStoreTrainSplit, ...
    'DataAugmentation', imgAugment, ...
    'ColorPreprocessing', 'gray2rgb');

augVal = augmentedImageDatastore(inputSize, imgStoreVal, ...
    'ColorPreprocessing', 'gray2rgb');

augTest = augmentedImageDatastore(inputSize, imgStoreTest, 'ColorPreprocessing', 'gray2rgb');

%% model architecture

layers = [
    imageInputLayer([224 224 3], 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv_4')
    batchNormalizationLayer('Name', 'bn_4')
    reluLayer('Name', 'relu_4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_4')

    fullyConnectedLayer(256, 'Name', 'fc_1')
    reluLayer('Name', 'relu_fc')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(classNum, 'Name', 'fc_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

%% options for performance adjustments

options = trainingOptions('adam', ...   % optimizer
    'ExecutionEnvironment', 'gpu', ...  % device adjustment
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'MaxEpochs', epochNum, ...
    'InitialLearnRate',1e-4, ...        % learning rate => 0.0001
    'MiniBatchSize', 64, ...          
    'Shuffle','every-epoch', ...        %
    'Verbose', false, ...               % 
    'Plots','training-progress');       % opens a window to observe training process


%% start the training
trainedNet = trainNetwork(augTrain, layers, options);

%% evaluating the model (direc)
YPred = classify(trainedNet, augTest);
YTrue = imgStoreTest.Labels;

accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

figure;
confChart = confusionchart(YTrue, YPred, 'Normalization', 'total-normalized');
confChart.Title = 'Confusion Matrix (Global Prediction Distribution)';
confChart.CellLabelFormat = '%.0f';

%% storing the model and appropriate information
save('trainedBirdModel.mat', 'trainedNet', 'options');

end
