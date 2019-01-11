clear all
train = csvread('./CVRR_splined_train_data.csv');
train_label = csvread('./CVRR_splined_train_label.csv');
test = csvread('./CVRR_splined_test_data.csv');
test_label = csvread('./CVRR_splined_test_label.csv');

%test_label = csvread('CVRR_splined_test_label.csv');
%% 

dimension = 50;
num_test = 500;
N = 49;
M = num_test;

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
%% CVX
turns = 10;
time = zeros(turns,1);
errors = zeros(turns,M);
for turn = 1:turns
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
    %disp(turn)
    
    %Y = Y_raw(:,(M * (turn - 1) + 1): (M * turn));
    t = ones(T,1);
    m = ones(M,1);
    n = eye(N);
    mm = eye(M);
    % e = rand;
    e = 0.1;
    l = 0.5;
    tic
    cvx_begin quiet
        variables S(T,M) Z(T,M) W1(T,T) W2(M,M);
        minimize( 0.5*(trace(W1)+trace(W2))+l*t'*Z*m);
        subject to
            [W1 S;S' W2] >= 0;
            -Z <= S <= Z;
            [e*n (Y-A*S);(Y-A*S)' e*mm] >= 0;
    cvx_end
    fprintf('time at %d: \n',num_train);
    time(turn)=toc;
    % Evaluation
    tmp = A*S;
    error = [];
    for i = 1:M
        error = [error, norm(tmp(:,i) - Y(:,i), 2) / norm(Y(:,i), 2)];
    % e = sum((tmp-Y).^2)./sum(Y.^2);
    end
    errors(turn,:)=error;
%     % % Check accuracy
%     predict_label = zeros(1, M);
%     for i = 1:M
%         if error(i) > 0.82
%             predict_label(i) = 1;
%         else
%             predict_label(i) = 0;
%         end
%     end
%     %label_true = [zeros(1,M/2), ones(1,M/2)];
%     % label_true = test_label(M * (turn - 1) + 1: (M * turn));
%     index_true = find(label_true == predict_label);
%     accuracy = length(index_true)/M;
%     fprintf('accuracy: %f \n',accuracy);
end

