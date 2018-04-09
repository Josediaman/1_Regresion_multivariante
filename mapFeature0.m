function out = mapFeature0(X1, d)
% out: Polinomic parameteres.
% X1: First variable. 
% d: Degree of the polynomic.


degree = d;
out = X1(:,1);
for i = 2:degree
        out(:, end+1) = (X1.^(i));
end


end