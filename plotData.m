function plotData(x, y, theta)
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% theta: Parameters of the regresion.


x_plot=x(:,2);
y_pred=x*theta;


figure; 
plot(x_plot,y,'rx','MarkerSize',10);
xlabel('x');
ylabel('y');


hold on;
plot(x_plot, y_pred,'bo',10);
legend('Training data', 'Regression');
hold off;


end
