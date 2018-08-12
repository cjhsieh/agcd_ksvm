%[y X] = libsvmread('../data/ijcnn1.tr');
clear all
load ijcnn_data
rand('seed',0);
minimum = -1.391082908935821e+03;
%minimum = -1.326123152952617e+03;
%aaa = randperm(size(X, 1));
aaa = 1:size(X,1);
n = 1000;
ytest = y(aaa(n+1:n*2), 1);
Xtest = X(aaa(n+1:n*2),:);

y = y(aaa(1:n), 1);
X = X(aaa(1:n), :);
maxiter = 20000;
gamma = 2;
c = 32;
Q = rbf(X, X, gamma).*(y*y');
n = size(X,1);
Qdiag = ones(n, 1);
alpha = zeros(n, 1); 
grad = zeros(n, 1)-1;

%% non-accelerated version
for iter = 1:maxiter
	%% select working set
	[max_val, j] = max(  abs(alpha - max(min(alpha - grad, c), 0) ) );
	%% update
	alphanew = min(c, max(0, alpha(j) - grad(j)/Qdiag(j)));
	delta_alpha = alphanew - alpha(j);
	grad = grad + delta_alpha*Q(:,j);
	alpha(j) = alpha(j) + delta_alpha;
	obj(iter) = 0.5*grad'*alpha - 0.5*sum(alpha);
end
figure
plot(log10(obj-minimum), 'k')

%% Non-strongly convex version
x = zeros(n, 1); 
x_grad = zeros(n, 1)-1;
y = x;
y_grad = x_grad;
z = x;
z_grad = x_grad;
theta = 1;
t = 100;

for iter = 1:maxiter
	y = (1-theta)*x + theta*z;
	y_grad = (1-theta)*x_grad + theta * z_grad; %% in svm-dual gradient is a linear function
	%% select working set
	[max_val, j] = max(  abs(y - max(min(y - y_grad, c), 0) ) );
	%% update
	xjnew = min(c, max(0, y(j) - y_grad(j)/Qdiag(j))); %% closed form solution for coordinate update
	delta = xjnew - y(j);
	x_grad = y_grad + delta*Q(:,j);
	x = y;
	x(j) = y(j) + delta;
	obj(iter) = 0.5*x_grad'*x - 0.5*sum(x);

	%% Assume the second sequence uses the same j
	znew = min(c, max(0, z(j) - y_grad(j)/theta/t/Qdiag(j)));
	delta_z = znew - z(j);
	z_grad = z_grad + delta_z*Q(:,j);
	z(j) = z(j) + delta_z;

% 	theta = (-1+sqrt(1+4*theta^2))/2.0;
    theta = 2/(iter+1);
end
min(obj)
hold on
plot(log10(obj-minimum), 'r')
% temp = min(obj)
%Qtest = rbf(Xtest, X, gamma);
%pred_val = Qtest*(alpha.*y);
%acc = sum((pred_val.*ytest)>0)/nnz(ytest)

%% Strongly convex version
x = zeros(n, 1); 
x_grad = zeros(n, 1)-1;
y = x;
y_grad = x_grad;
z = x;
z_grad = x_grad;
u = x;
u_grad = x_grad;
% mu = min(eig(Q))/max(diag(Q));
mu = 0.01;
t = 100;
a = sqrt(mu)/(t+sqrt(mu));
b = mu*a/t/t;

for iter = 1:maxiter
	y = (1-a)*x + a*z;
	y_grad = (1-a)*x_grad + a *z_grad; %% in svm-dual gradient is a linear function
	%% select working set
	[max_val, j] = max( abs(y - max(min(y - y_grad, c), 0) ) );
	%% update
	xjnew = min(c, max(0, y(j) - y_grad(j)/Qdiag(j))); %% closed form solution for coordinate update
	delta = xjnew - y(j);
	x_grad = y_grad + delta*Q(:,j);
	x = y;
	x(j) = y(j) + delta;
	obj(iter) = 0.5*x_grad'*x - 0.5*sum(x);

	%% Assume the second sequence uses the same j
    u = a^2/(a^2+b)*z + b/(a^2+b)*y;
    u_grad = a^2/(a^2+b)*z_grad + b/(a^2+b)*y_grad;
	znew = min(c, max(0, u(j) - a/(a^2+b)/t*y_grad(j)/Qdiag(j)));
	delta_z = znew - u(j);
    z = u;
    z(j) = u(j) + delta_z;
	z_grad = u_grad + delta_z*Q(:,j);

% 	theta = (-1+sqrt(1+4*theta^2))/2.0;
%     theta = 2/(iter+1);
end
minimum = min(obj)
plot(log10(obj-minimum), 'b')
print -dpng tmp.png
hold off
