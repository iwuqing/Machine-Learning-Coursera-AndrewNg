function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% 参考选择值
reference_value = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% 预测误差，初始化为0
predict_error = 0;

for i = 1:8
    % 临时C
    temp_C = reference_value(i);
    for j = 1:8
        % 临时sigma
        temp_sigma = reference_value(j);
        % 训练模型
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        % 预测
        predictions = svmPredict(model, Xval);
        temp_error = mean(double(predictions~=yval));
        if((j == 1 && i == 1) || predict_error > temp_error)  % 第一次预测，直接初始化参数
            C = temp_C;
            sigma = temp_sigma;
            predict_error = temp_error;
        end
    end
end




% =========================================================================

end
