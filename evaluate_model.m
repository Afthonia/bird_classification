function evaluate_model(trainedModel, filePath)
    
    arguments
        trainedModel (1,:) string {} = "trainedBirdNet.mat"
        filePath (1,:) string {} = "bird1.png"
    end

    %% using the trained model
    
    load(trainedModel, 'trainedNet');
    
    fullPath = "eval_img/" + filePath;
    bird = imread(fullPath);
    
    disp("Loading the image from: " + fullPath);
    %disp(size(bird));

    bird = imresize(bird, [224 224]);
    
    % checking whether the image is 3d rgb or grayscale
    if size(bird, 3) == 1
    
        %if grayscale, turn it to rgb by copying the single layer 3 times
        bird = repmat(img, 1, 1, 3);
    end
    
    prediction = classify(trainedNet, bird);
    
    disp("Predicted Bird Class: " + string(prediction));
end

