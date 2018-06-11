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
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hypothesis = X * theta;
thetaUnRegKnot = theta;
thetaUnRegKnot(1) = 0;
regularization = (lambda / (2 * m)) * sum((thetaUnRegKnot .^ 2));
cost = (1 / (2 * m)) * sum((hypothesis - y) .^ 2);
J = cost + regularization;

%we can compute for each theta then for all but the first we can add reg
%grad = 1 / m * X' * (hypothesis - y); 
%we then reg from the second element remembet j = 0, we have theta1 set to zero so we shlould be fine with the whole thing
%grad(2 : size(theta)) = grad(2 : size(theta)) + (lambda / m) * theta(2 : size(theta));
%we can do this in one go as j >=1 the zero theta val will make reg param zero
grad = 1 / m * X' * (hypothesis - y)  + ((lambda / m) * thetaUnRegKnot);
% =========================================================================

grad = grad(:);

end
