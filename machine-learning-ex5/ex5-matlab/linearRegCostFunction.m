function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); %n*1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X(m*n+1) theta(n+1*1) y(m*1)
temp_theta = theta(2:end, :);
J = (X*theta - y)'*(X*theta - y)./(2*m) + (lambda/(2*m)).*temp_theta'*temp_theta;

grad(1) = (1/m).*X(:, 1)'*(X*theta - y);
grad(2:end) = (1/m).*X(:, 2:end)'*(X*theta - y) + (lambda/m).*temp_theta'*temp_theta;

% =========================================================================

grad = grad(:);

end
