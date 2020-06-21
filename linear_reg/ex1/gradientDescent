function [theta, J_history, theta0_history, theta1_history] = gradientDescent(X, y, theta, alpha, iterations)

m = length(y); % number of training examples
J_history = zeros(iterations, 1); % lines 4-6 can visualize steps if it's single linear regression
theta0_history = zeros(iterations, 1); 
theta1_history = zeros(iterations, 1);
new_theta = zeros(length(theta), 1);

for iter = 1:iterations
    for i = 1:length(theta)
        
        gradient = (1/m)*transpose(X*theta - y)*X(:, i); % applicable to multi linear regression
        new_theta(i) = theta(i) - alpha*(gradient);
        
    end
    
    theta = new_theta;
    J_history(iter) = computeCost(X, y, theta);
    theta0_history(iter) = theta(1);
    theta1_history(iter) = theta(2);

end

end
