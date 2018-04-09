function [J, grad] = costFunctionReg(theta, X, y,lambda)
% J: Cost of the regresion with theta.
% grad: Gradient of J.
% theta: Parameters of the regresion.
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% lambda: Paramer of the regularization.


m = length(y); 
J = 0;
grad = zeros(size(theta));
theta2=theta(2:size(theta));

predictions=X*theta;
Errors=(predictions-y);
J=(1/(2*m))*(Errors'*Errors)+(lambda/(2*m))*theta2'*theta2;

grad(1)=(1/m)*sum(Errors.*X(:,1))';
grad(2:size(theta))=(1/m)*sum(Errors.*X(:,2:size(theta)))'+(lambda/m)*theta2;


end
