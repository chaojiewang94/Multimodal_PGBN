function [avg_prec, std_recall] = trec_avg_precision(relevance)
% compute trec-style average precision at standard recall points
%
% Reference: Introduction to Information Retrieval 
%            By Christopher D. Manning, Prabhakar Raghavan & Hinrich Schütze
%            Section: Evaluation of ranked retrieval results
%            Website: http://informationretrieval.org/
%

% compute precision and recall
N = size(relevance, 2);
cum_rlv = cumsum(relevance, 2);
precision = bsxfun(@rdivide, cum_rlv, 1:N);
recall = bsxfun(@rdivide, cum_rlv, cum_rlv(:,end));

numLevels = 101;
% compute average precision at standard recall points
avg_prec = zeros(1, numLevels);
std_recall = linspace(0, 1, numLevels);
for i = 1:numLevels
    precision(recall < std_recall(i)) = -Inf;
    avg_prec(i) = mean(max(precision, [], 2));
end
