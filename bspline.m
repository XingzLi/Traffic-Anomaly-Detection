close all;
clear all;
processed_data = [];
dataset = 'CROSS';

datadir = sprintf('/Users/yimeng/Documents/2018spring/Sparsity/Project/data/CVRR_dataset_trajectory_analysis_v0/%s', dataset);
filename = sprintf('%s/train.mat', datadir);
load(filename);
n = 50;
for i = 1:length(tracks_train)
    x = tracks_train{i}(1,:);
    y = tracks_train{i}(2,:);
    minx = min(x);
    maxx = max(x);
    xx = minx:((maxx - minx)/ (n - 1)):maxx;
    yy = spline(x, y, xx);
    processed_data = [processed_data; xx];
    processed_data = [processed_data; yy];
end
eliminated_train_data = [];
label = [];
for i = 1:17
    index_i = find(labels_train == i);
    for j = index_i
        eliminated_train_data = [eliminated_train_data; processed_data(2 * j - 1,:)];
        eliminated_train_data = [eliminated_train_data; processed_data(2 * j,:)];
        label = [label; i];
    end
end

index_i = find(labels_train == 19);
for j = index_i
    eliminated_train_data = [eliminated_train_data; processed_data(2 * j - 1,:)];
    eliminated_train_data = [eliminated_train_data; processed_data(2 * j,:)];
    label = [label; 19];
end

size(eliminated_train_data)
csvwrite('/Users/yimeng/Documents/2018spring/Sparsity/Project/data/CVRR_splined_train_data.csv',eliminated_train_data)
csvwrite('/Users/yimeng/Documents/2018spring/Sparsity/Project/data/CVRR_splined_train_label.csv',label)