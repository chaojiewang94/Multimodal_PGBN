%% 根据数据格式名字选择数据集
switch DataName
    
    
    case 'Multilabel1'
% 25k个样本 15k训练 10k测试
        load('bibtex_new.mat');
        image_labeled_feature = image_labeled_without_sift';   %只用了1857维度特征
        text_labeled_feature  = text_labeled'; 
        
%  对应splits 和 epoch
        splits_index = 1;
        epoch_index = 1;
        
        prepar.trindx  =  flicker_splits{splits_index}.tr_index; % 25000 中训练的index
        prepar.teindx  =  flicker_splits{splits_index}.te_index; % 25000 中测试的index
        prepar.splits_index = splits_index;
        prepar.epoch_index  = epoch_index;
        
        
    case 'flicker_25k'
% 25k个样本 15k训练 10k测试
        load('flicker_labeled_split_10_epoch_5.mat');
        image_labeled_feature = image_labeled_without_sift';   %只用了1857维度特征
        text_labeled_feature  = text_labeled'; 
        
%  对应splits 和 epoch
        splits_index = 1;
        epoch_index = 1;
        
        prepar.trindx  =  flicker_splits{splits_index}.tr_index; % 25000 中训练的index
        prepar.teindx  =  flicker_splits{splits_index}.te_index; % 25000 中测试的index
        prepar.splits_index = splits_index;
        prepar.epoch_index  = epoch_index;
        
        
    case 'flicker_25_w10'
%  1995个样本 1500训练 495测试 每个样本词数>=10
%  check by chaos 2017/5/23
        
        load('labeled_w10.mat');
        image_labeled_feature = labeled_feature_w10'; %D*N
        text_labeled_feature  = labeled_tags__w10'; 

%  对应splits 和 epoch
        splits_index = 1;
        epoch_index = 1;
        
        prepar.trindx = 1:1500;
        prepar.teindx = 1501:1995;
        prepar.splits_index = splits_index;
        prepar.epoch_index  = epoch_index;
        
end


%% 根据格式需要处理数据格式
switch DataType
    
    case 'Postive'
%%  特征维度高斯归一，再加上每个维度的最小值，保证是postive的
%  check by chaos 2017/5/23
%  处理前数据形式  图像 D*N
%  处理后数据形式  图像 D*N        
        image_labeled_norm = normalization_2(image_labeled_feature',3); % mode =3为高斯归一  N*D
        image_labeled_norm = image_labeled_norm';                      % D*N
        image_labeled_norm_min = min(image_labeled_norm,[],2);
        image_labeled_process = image_labeled_norm - repmat(image_labeled_norm_min,1,size(image_labeled_norm,2));
        X_img_all   = sparse(double(image_labeled_process));
        
%  处理前数据形式  文本 D*N
%  处理后数据形式  文本 D*N  
        X_tags_all = sparse(double(text_labeled_feature));        % 总共25000个               

        
    case 'Count'
%  特征维度归一，离散化到0-25
%  check by chaos 2017/5/23
%  处理前数据形式  图像 D*N
%  处理后数据形式  图像 D*N                     
        image_labeled_min = min(image_labeled_feature,[],2);
        image_labeled_max = max(image_labeled_feature,[],2);
        image_labeled_norm = (image_labeled_feature - repmat(image_labeled_min,1,size(image_labeled_feature,2)))...
                        ./repmat(image_labeled_max-image_labeled_min,1,size(image_labeled_feature,2));
        image_labeled_process = round(image_labeled_norm*25);
        X_img_all   = sparse(double(image_labeled_process));
        
%  处理前数据形式  文本 D*N
%  处理后数据形式  文本 D*N  
        X_tags_all = sparse(double(text_labeled_feature*150));        % 总共25000个               
  
        
    otherwise
        error('Wrong "ToBeAnalized"')
end

clearvars -EXCEPT X_img_all X_tags_all prepar DataName DataType dataname ToBeAnalized SystemInRuningLinux SuPara Settings;