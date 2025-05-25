function visualizeDataset(imgStore)
    figure;
    perm = randperm(length(imgStore.Files), 16);  % Show 16 random images
    for i = 1:16
        subplot(4,4,i);
        img = readimage(imgStore, perm(i));
        imshow(img);
        title(string(imgStore.Labels(perm(i))));
    end
    sgtitle('Sample Training Images');
end

