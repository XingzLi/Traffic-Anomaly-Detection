clear all
train = csvread('./CVRR_splined_train_data.csv');
train_label = csvread('./CVRR_splined_train_label.csv');
test = csvread('./CVRR_splined_test_data.csv');
test_label = csvread('./CVRR_splined_test_label.csv');

%test_label = csvread('CVRR_splined_test_label.csv');
%% 
% % metadata
% % Set parameters lambda, mu
al = 0.9;
la = 0.01;
% lambda1 = 0.1 * ones(n, 1);
% lambda2 = 0.1 * ones(n, 1);

mu = 0.1;
max_iteration = 10000;
max_iteration_admm = 1000;
epsilon = 0.1;
sigma_ori = 0.1;
epsilon_admm = 0.1;

dimension = 50;
num_test = 500;
N = 49;
M = num_test;
m = 49;
% n = 500;
% % Angle
% A = zeros(dimension - 1,num_train);
% for i = 1:num_train
%     angle = zeros(1, dimension - 1);
%     x  = train(2 * i - 1, :);
%     y = train(2 * i, :);
%     for j = 1:dimension - 1
%         angle(1, j) = sin(atan((y(j + 1) - y(j))/(x(j + 1) - x(j))));
%     end
%     A(:, i) = angle';
% end
Y = [];
normal_index = find(test_label ~= -1);
for i =1:num_test / 2
    x = test(2 * normal_index(i + 35) - 1,:);
    y = test(2 * normal_index(i + 35),:);
    angle = zeros(1,dimension - 1);
    for j = 1:dimension - 1
        angle(1,j) = sin(atan((y(j + 1) - y(j)) / (x(j + 1) - x(j))));
    end
    Y(:,i) = angle';
end
abnormal_index = find(test_label == -1);
for i =1:num_test / 2
    x = test(abnormal_index(i + 2) * 2 - 1,:);
    y = test(abnormal_index(i + 2) * 2,:);
%     x = test(abnormal_index(length(abnormal_index)-i) * 2 - 1,:);
%     y = test(abnormal_index(length(abnormal_index)-i) * 2,:);

    angle = zeros(1,dimension - 1);
    for j = 1:dimension - 1
        angle(1, j) = sin(atan((y(j + 1) - y(j)) / (x(j + 1) - x(j))));
    end
    Y(:,num_test/2 + i) = angle';
end
label_true = [zeros(1,M/2), ones(1,M/2)];
turns = 10;
time = zeros(turns,1);
errors = zeros(turns,M);
accu = zeros(turns,M);
for turn = 1:turns
    turn
    num_train = (turn-1)*50+100;
    T = num_train;
    av = floor(num_train/18);
    A=zeros(N,T);
    limit=av*ones(1,19);
    j = 0;
    for i = 1:length(train_label)
        if limit(train_label(i))~= 0 && train_label(i)~=18
            limit(train_label(i)) = limit(train_label(i))-1;
            x=train(i*2-1,:);
            y=train(i*2,:);
            j = j+1;
            angle = zeros(1, N);
            for k = 1:N
                angle(1, k) = sin(atan((y(k + 1) - y(k))/(x(k + 1) - x(k))));
            end
            A(:, j) = angle';
        end
    end
    th = num_train-j;
    ind = ceil(rand(th,1)*100);
    for i = 1:th
        x=train(ind(i)*18-1,:);
        y=train(ind(i)*18,:);
        j = j+1;
        angle = zeros(1, N );
        for k = 1:N
            angle(1, k) = sin(atan((y(k + 1) - y(k))/(x(k + 1) - x(k))));
        end
        A(:, j) = angle';
    end
    tic;
    % % Initialize X, S, Z
    n = 100 + 50 * (turn - 1);
    t = 500;
    lambda1 = 0.1 * ones(n, t);
    lambda2 = 0.1 * ones(n, t);
    X = random('Normal',0,1,n,t);
    S = X;
    Z = X;
    % % Update 
    ad = 0;
    while ad < max_iteration_admm
        ad = ad + 1;
        % % Update X, S 
        k = 0;
        while k < max_iteration
%             disp(k)
            x_prev = X;
            s_prev = S;
            xG = x_prev - al * (lambda1 + mu * (x_prev - Z));
            sG = s_prev - al * (lambda2 +  mu * (s_prev - Z));
            S = Soft(al * la, sG);
            [U, Sigma, V] = svd(xG);
            X = U * Soft(al, Sigma) * V';
            k = k + 1;
            if abs(h(X,S,Z,mu,la,lambda1,lambda2) - h(x_prev,s_prev,Z,mu,la,lambda1,lambda2)) <= epsilon
                break;
            end     
        end
        % % Update Z
        z_prev = Z;
        z_left = sigma_ori * A' * A + 2 * mu * eye(n,n);
        z_right = sigma_ori * A' * Y + lambda1 + lambda2 + mu * (X + S);
        Z = inv(z_left) * z_right;
        % % Update lambda
        lambda1 = lambda1 + mu * (X - Z);
        lambda2 = lambda2 + mu * (S - Z);
        disp(norm([X - Z ; S - Z]))
        if norm([X - Z ; S - Z]) < epsilon_admm
            break
        end
    end
    tmp = A*S;
    error = zeros(1, t);
    for i = 1:t
        error(i) = norm(tmp(:,i) - Y(:,i), 2) / norm(Y(:,i), 2);
    end
%     plot(1:t,error);
%     disp('please enter: thre = ?')
%     predict = zeros(t, 1);
%     for check = 1:t
%         if error(check) < thre
%             predict(check) = 0;
%         else
%             predict(check) = 1;
%         end
%     end
% %     plot(1:t, error);
%     difference = predict - [zeros(t / 2,1); ones(t / 2,1)];
%     accuracy = size(find(difference == 0), 1) / t;
    time_record(turn) = toc;
end


function [result] = Soft(c, X)
    for i = 1:size(X,1)
        for j = 1:size(X,2)
            if abs(X(i,j)) < c
                X(i,j) = 0;
            elseif X(i,j) >= c
                X(i,j) = X(i,j) - c;
            else
                X(i,j) = X(i,j) + c;
            end
        end
    end
    result = X;
end

function [value] = h(X,S,Z,mu,la,lambda1,lambda2)
    value = mu / 2 * norm(2 * Z - X - S, 'fro') ^ 2 + trace(lambda1' * (X - Z) +lambda2' * (S - Z)) + sum(svd(X)) + la * norm(S, 1);
end