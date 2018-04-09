
%% ................................................
%% ................................................
%%    LINEAR REGRESSION WITH MULTIPLE VARIABLES
%% ................................................
%% ................................................





%% 1. Clear and Close Figures
clear ; close all; clc





%% ================ Part 1: Date ================
fprintf('\n \nDATE\n.... \n \n \n');   





%% 2. Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add your own file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Loading data ...\n');   
%%%%%%********Select number of features********
num_feat=2;                
%%%%%%********Select archive********   
data = load('ex1data1.txt');  
X = data(:, 1:num_feat-1);   
y = data(:, num_feat);
fprintf('(X,y) (10 items)\n');   
[X(1:10,:) y(1:10,:)]
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 2'. No lineal regression 
%%%%% ************* Select grade of polynomial ***********
poly_grade=6;
X=mapFeature0(X, poly_grade);


%% 3. Normalizing Features and adding first colum of ones
fprintf('Normalizing Features and adding first colum of ones ...\n');
[X mu sigma] = featureNormalize(X);
fprintf('X (normal) (10 items)\n');
[m, n] = size(X);   
X = [ones(m, 1) X];
X(1:10,:)
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 4. Select train, cross and test validation sets
[X, y, Xval, yval, Xerr, yerr, m, n] = ...
    selectsets(X, y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract sets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% ======= Part 2: Multiple regression====
fprintf('REGULARIZED MULTIPLE REGRESSION\n........................................... \n \n \n \n');





%% 4. Initial values
%%%%% *************Select initial theta and lambda***********
initial_theta = zeros(n, 1);
lambda = 0;
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Initial values: \n\n');
fprintf('Theta: \n');
fprintf(' %f \n', initial_theta);
fprintf('\n');
fprintf('Cost: \n');
fprintf(' %f \n', cost);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 5. Run Gradient Descent with octabe function (fminunc)
fprintf('Running gradient with octabe function...\n \n');
%%%%% *************Select iterations***********
num_iters = 5000;
options = optimset('GradObj', 'on', 'MaxIter', num_iters);
[theta, J] = ...
         fminunc (@(t)(costFunctionReg(t, X, y, lambda)), ...
                initial_theta, options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract theta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optional: Execute your own gradient descent.

%fprintf('Running gradient descent with alpha ... \n \n ');
%%%%% *************Select iterations***********
%num_iters = 1000;
%[theta, J_his] = gradientDescentMulti(X, y, theta, alpha, %num_iters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



[cost_train, grad0] = costFunctionReg(theta, X, y, 0);
[cost_cross, grad0] = costFunctionReg(theta, Xval, yval, 0);
[cost_error, grad0] = costFunctionReg(theta, Xerr, yerr, 0);

%% 6. Display results
fprintf('Theta: \n');
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('Cost: \n');
fprintf(' %f \n', J);
fprintf('\n');
fprintf('Cost train: \n');
fprintf(' %f \n', cost_train);
fprintf('\n');
fprintf('Cost cross: \n');
fprintf(' %f \n', cost_cross);
fprintf('\n');
fprintf('Cost test: \n');
fprintf(' %f \n', cost_error);
fprintf('\n');

fprintf('\nProgram paused. Press enter to continue.\n \n \n \n');
pause;




%% ================ Part 2': GRAPHIC ================
fprintf('GRAPHIC \n...... \n \n \n \n');

fprintf('\nCheck the graphic\n \n');


plotData(X, y, theta);






%% ================ Part 3: Sample to predict ================
fprintf('SAMPLE\n...... \n \n \n \n');





%% 7. Select a sample to predict
%%%%% *************Select sample to predict***********
x1 = X(6,:);                   
x2 = (x1(1,2:end).*sigma).+mu;


%% 8. Estimate the y of the sample
estimation_y = x1*theta; 
fprintf('Prediction of the sample:\n x_pred= ');
fprintf('%f  ',x2(1,:));
fprintf('\n y_pred= %f \n \n',estimation_y);
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ==== Part 4: Learning Curve for Linear Regression ========
fprintf('\n\nLEARNING CURVE\n.............. \n \n \n \n');




close all; 
[error_train, error_val] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  lambda, initial_theta, options,poly_grade);

d=poly_grade+1;
plot(d:m, error_train, d:m, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Check if there is a bios or variation problem.\n\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;






%% ================ Part 5: Validation ================
fprintf('\n\nVALIDATION\n.......... \n \n \n \n');






[lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, initial_theta, options);


figure;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Error Train', 'Error Cross');
xlabel('lambda');
ylabel('Error');
fprintf('\nlambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end


fprintf('\n Actual lambda: \n');
fprintf(' %f \n', lambda);
fprintf('\nThe best lambda has the lowest validation error.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;







