function [ traindata,testdata ] = normalization( traindata,testdata,mode )
%UNTITLED3 Summary of this function goes here
%   traindata、testdata:N*D矩阵：N为样本个数，D为单个样本的特征维度
switch mode
    case 0   %%不归一化
        
    case 1   %%样本强度归一
        temp=traindata';
        temp1=testdata';
        traindata=traindata./repmat((max(temp))',1,size(traindata,2));
        testdata=testdata./repmat((max(temp1))',1,size(traindata,2));
    case 2      %%样本能量归一
        traindata=traindata./repmat((sqrt(sum(traindata'.^2)))',1,size(traindata,2));
        testdata=testdata./repmat((sqrt(sum(testdata'.^2)))',1,size(traindata,2));
    case 3      %%特征维度标准化高斯归一
        [traindata,mu,sigma]=zscore(traindata);
        testdata=bsxfun(@minus,testdata,mu);
        testdata=bsxfun(@rdivide,testdata,sigma);
    case 4      %%能量归一之后再减均值
        traindata=traindata./repmat((sqrt(sum(traindata'.^2)))',1,size(traindata,2));
        testdata=testdata./repmat((sqrt(sum(testdata'.^2)))',1,size(traindata,2));
        mu=mean(traindata);
        traindata=bsxfun(@minus,traindata,mu);
%         mu=mean(testdata);
        testdata=bsxfun(@minus,testdata,mu);
end

