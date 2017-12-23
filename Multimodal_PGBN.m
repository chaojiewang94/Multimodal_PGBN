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

clear,clc,close all;


%% load flicker Data
DataName    =  'Multilabel1';        %% choose input datatset: 'flicker_25k'  'flicker_25_w10'
%% 'Multilabel'
DataType    =  'Count';              %% choose input datatype: 'Count' or 'Postive' 

addpath(genpath('./'));
load_flicker_data;

DataName = [DataName,'_s',num2str(prepar.splits_index),'_e',num2str(prepar.epoch_index)];

%%  PGBN params
K = [500 200]  ;   T   =   length(K)   ;                                %% set structure of your networks here [1024] or [1024 1024] in our paper
SuPara.ac      =   1      ;   SuPara.bc      =   1   ;
SuPara.a0pj    =   0.01   ;   SuPara.b0pj    =   0.01   ;
SuPara.e0cj    =   1      ;   SuPara.f0cj    =   1   ;
SuPara.e0c0    =   1      ;   SuPara.f0c0    =   1    ;
SuPara.a0gamma =   1      ;   SuPara.b0gamma =   1   ;
SuPara.eta     =   0.01 * ones(1,T)     ;

%% Framework Settings

Settings.GibbsInit      =   0   ; 
Settings.UpdateRGamma   =   0   ;


%% GBN Settings
GibbsSampler = 'Multi_PGBN_Generate_Tags';   %% Multi_PGBN_Burin_Collect'
Date = '2017_12_18';
Settings.DataName = DataName;
Settings.DataType = DataType;
Settings.Date = Date;
Settings.GibbsSampler = GibbsSampler;
Settings.IterAll    =   200    ;             %% set the total iterations here

%% Batch Gibbs
if 1
    for trial   =    2              %1 : 5
        rng(trial,'twister');
        switch GibbsSampler
            case 'Multi_PGBN_Burin_Collect'
                Multi_PGBN_Gibbs_Burin_Collect(X_img_all,X_tags_all,prepar,K,T,trial,DataType,DataName,SuPara,Settings);
            case 'Multi_PGBN_Generate_Tags'
                Multi_PGBN_Generate_Tags(X_img_all,X_tags_all,prepar,K,T,trial,DataType,DataName,SuPara,Settings);
        end
    end
end



