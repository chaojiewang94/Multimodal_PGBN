clc
clear
close all


addpath(genpath('D:\Íõ³¬½Ü\PGBN_wcj\flicker_data'))
load image_labeled.mat
load text_labeled.mat
load label_of_25K.mat

num_split = 10;
flicker_splits = cell(1,num_split);
for splits_index = 1:num_split
    flicker_splits{splits_index}.tr_te_index = [randperm(25000)]';
    flicker_splits{splits_index}.tr_index = flicker_splits{splits_index}.tr_te_index(1:15000);
    flicker_splits{splits_index}.te_index = flicker_splits{splits_index}.tr_te_index(15001:25000);
    
    cv_num = 15000/5;
    for epoch_index = 1:5
        tmp_index = [1:15000]';
        cv_te_index_in_tmp = (epoch_index-1)*cv_num+1 : (epoch_index)*cv_num;
        tmp_index(cv_te_index_in_tmp)=0;
        cv_tr_index_in_tmp = find(tmp_index~=0);
        
        flicker_splits{splits_index}.cv_tr_index(:,epoch_index) =     flicker_splits{splits_index}.tr_index(cv_tr_index_in_tmp);
        flicker_splits{splits_index}.cv_te_index(:,epoch_index) =     flicker_splits{splits_index}.tr_index(cv_te_index_in_tmp);
    end  
end
    