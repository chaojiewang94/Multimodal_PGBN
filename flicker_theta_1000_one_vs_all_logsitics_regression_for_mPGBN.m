% =======================================================================
% Mutlimodal Poisson Gamma Belief Network
% =======================================================================
% 
% This is the code for the paper "Mutlimodal Poisson Gamma Belief Network" presented at AAAI 2018 
% 
% Created by Chaojie Wang , Bo Chen at Xidian University and Mingyuan Zhou at University of Texas at Austin
% 
% =======================================================================
% LICENSE
% =======================================================================
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% Copyright (c), 2018, Chaojie Wang 
% xd_silly@163.com

%% load data
addpath(genpath('./'))
Theta_tr_joint = Theta_joint_collect{1,1};        % D*N
Theta_te_joint = Theta_joint_Test_collect{1,1};   % D*N
tr_index = prepar.trindx;
te_index = prepar.teindx;

load flicker_labeled_split_10_epoch_5.mat

%% =============one-vs-all logistic regression=============
MAP = zeros(38,1);
Presion = zeros(38,1);

[Theta_tr_joint,Theta_te_joint] = normalization(Theta_tr_joint',Theta_te_joint',3);
Theta_tr_joint = Theta_tr_joint';
Theta_te_joint = Theta_te_joint';
relevance = zeros(38,length(te_index));

nn = nnsetup([size(Theta_tr_joint,1),38]);
nn.momentum         = 0.0;          %  Momentum
nn.learningRate     = 0.01;
nn.weightPenaltyL2  = 0.1; 
nn.output           = 'sigm';
opts.numepochs      = 200;          %  Number of full sweeps through data
opts.batchsize      = 200;          %  Take a mean gradient step over this many samples

[nn, L] = nntrain(nn,Theta_tr_joint' , label_of_25k(tr_index,:) , opts);
nn = nnff(nn,Theta_te_joint',label_of_25k(te_index,:));  
z_1 = nn.a{1}*nn.W{1}';

%% =============caculate MAP and Pre@50=============
for label_index = 1:38
    
    label_ground_truth = label_of_25k(te_index,label_index);
    label_predict = z_1(:,label_index);
    label_pre_grd = [label_predict,label_ground_truth];
    [label_sort,label_index_sort] = sortrows(label_pre_grd,1);
    label_sort = flipud(label_sort);            % 从大到小排序
    relevance(label_index,:) = label_sort(:,2)';
    
    count = 0;
    precison_sum = 0;
    for index = 1:length(te_index)
        if (label_sort(index,2) == 1)
            count = count+1;
            precison_sum = precison_sum + count/index;
        end
    end
    MAP(label_index) = precison_sum/count;
    Presion(label_index) = sum(label_sort(1:50,2),1)/50;
    
end

disp(['Pre@50 is ',num2str(mean(Presion))]);

[avg_prec, std_recall] = trec_avg_precision(relevance);
disp(['MAP is ',num2str(mean(avg_prec))]);







 