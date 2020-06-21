% Compute cost function for any m number of linear regression variables 
function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = (1/(2*m))*transpose(X*theta - y)*(X*theta - y); % Vectorization
end
