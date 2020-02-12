clear all

%% Data Processing 
data = csvread('fashion-mnist_train.csv');
data_1 = csvread('fashion-mnist_test.csv');

%60,000 Training data available
Train_x = zeros(length(data),784); % Creating labels for Binary classification training
for i = 1: length(data)
    for j = 2:785
        Train_x(i,j-1) = data(i,j);   %Read training label y
    end
end

%10,000 Test data available
Test_x = zeros(length(data_1),784); % Creating labels for Binary classification training
for i = 1: length(data_1)
    for j = 2:785
        Test_x(i,j-1) = data_1(i,j);   %Read training label y
    end
end
Train_x = normalize(Train_x);
Test_x = normalize(Test_x);

%60,000 Training labels available
Train_y = zeros(1,length(data)); % Creating labels for Binary classification training
%only double class operations
for i = 1: length(data)
        Train_y(1,i) = int8(data(i,1))+1;   
end

%10,000 test labels available
Test_y = zeros(1,length(data_1)); % Creating labels for Binary classification test
for i = 1: length(data_1)
        Test_y(1,i) = int8(data_1(i,1))+1;  
end

%% Q5.2 _ a Multi-Class Online Learning Curve

% classify the pictures to its class labels
% I think the matlab arg max starts from 1,2,3,....,10 
% so we need to manipulate that
% result to start from 0,1,...,9
% it was simple to implement class [1,2,....,10] on the data it self

k = 10; d = 784; % #labels #features
w = zeros(k,d);  %intial vector for 784 features
% Training

T = 50;    %# of iterations
error_1 = zeros(1,T);   % Error count for each T Runs
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for ii = 1: length(Train_y)
        x = double(transpose(Train_x(ii,:))); 
        y = Train_y(1,ii);
        [a , y_p] = max(w*x);     %Dot product is w^T *x, I took care of the transpose in my variable organization
        
        if y ~= y_p 
            f_c = zeros(k,d); f_c(y ,:) = x;
            f_w = zeros(k,d); f_w(y_p ,:) = x;
            w = w + f_c - f_w;     
            count = count+1;
        end
    end
    error_1(1,i) = count;
end

% Multi-Class Classifications of Labels using PA


w = zeros(k,d);  %intial vector for 784 features
% Training

T = 50;    %# of iterations
error_2 = zeros(1,T);   % Error count for each T Runs
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for ii = 1: length(Train_y)
        x = double(transpose(Train_x(ii,:))); 
        y = Train_y(1,ii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
           count = count+1;
           f_c = zeros(k,d); f_c(y ,:) = x;
           f_w = zeros(k,d); f_w(y_p ,:) = x;
           loss = 1 - ( w(y,:)* x -   w(y_p,:)* x);    % Update logic?
           tau = loss/(norm(f_c - f_w)^2); 
           w = w + tau * (f_c - f_w); 
        end 
    end
    error_2(1,i) = count;
end
figure(1)
t = 1:T;
plot(t,error_2,'b--',t,error_1,'k--')
grid on
xlim([1 T])
ylim([0 max(error_2)+50])
xlabel('Iterations')
ylabel('Mistakes')
title('Multi-Class Online Learning curve')
legend('PA', 'Perceptron')


%% Q5.2 _ b Multi-Class training Accuracy

w = zeros(k,d);  %intial vector for 784 features
% Training

T = 20;    %# of iterations
error_training_1 = zeros(1,T);   % Error count for each T Runs
accuracy_training_1 = zeros(1,T);
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for ii = 1: length(Train_y)
        x = double(transpose(Train_x(ii,:))); 
        y = Train_y(1,ii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
            f_c = zeros(k,d); f_c(y ,:) = x;
            f_w = zeros(k,d); f_w(y_p ,:) = x;
            w = w + f_c - f_w;     
            count = count+1;
        end
    end
    error_training_1(1,i) = count;
    accuracy_training_1(1,i) = 1 - count/length(Train_y);
end

% Multi-Class Classifications of Labels using PA

w = zeros(k,d);  %intial vector for 784 features
error_training_2 = zeros(1,T);   % Error count for each T Runs
accuracy_training_2 = zeros(1,T);
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for ii = 1: length(Train_y)
        x = double(transpose(Train_x(ii,:))); 
        y = Train_y(1,ii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
           count = count+1;
           f_c = zeros(k,d); f_c(y ,:) = x;
           f_w = zeros(k,d); f_w(y_p ,:) = x;
           loss = 1 - ( w(y,:)* x - w(y_p,:)* x); 
           tau = loss/(norm(f_c - f_w)^2); 
           w = w + tau * (f_c - f_w); 
        end 
    end
    error_training_2(1,i) = count;
    accuracy_training_2(1,i) = 1 - count/length(Train_y);
end

figure(2)
t = 1:T;
plot(t,accuracy_training_2,'b--',t,accuracy_training_1,'k--')
grid on
xlim([1 T])
ylim([0.5 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Multi-Class training Accuracy curve')
legend('PA', 'Perceptron')


%% Q5.2 _ b Multi-Class test Accuracy

k = 10; d = 784; % #labels #features
w = zeros(k,d);  %intial vector for 784 features
% Training
error_test_1 = zeros(1,T);   % Error count for each T Runs
accuracy_test_1 = zeros(1,T);
for i = 1:T
    count = 0;        %Error count during Tth iteration
    
    for ii = 1: length(Train_y)
        x = double(transpose(Train_x(ii,:))); 
        y = Train_y(1,ii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
            f_c = zeros(k,d); f_c(y ,:) = x;
            f_w = zeros(k,d); f_w(y_p ,:) = x;
            w = w + f_c - f_w;  
        end
    end
    
    
    for iii = 1: length(Test_y)
        x = double(transpose(Test_x(iii,:))); 
        y = Test_y(1,iii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
            count = count + 1; 
        end
    end
    
    error_test_1(1,i) = count;
    accuracy_test_1(1,i) = 1 - count/length(Test_y);
end

% Multi-Class Classifications of Labels using PA

w = zeros(k,d);  %intial vector for 784 features
error_test_2 = zeros(1,T);   % Error count for each T Runs
accuracy_test_2 = zeros(1,T);
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for ii = 1: length(Train_y)
        x = double(transpose(Train_x(ii,:))); 
        y = Train_y(1,ii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
           f_c = zeros(k,d); f_c(y ,:) = x;
           f_w = zeros(k,d); f_w(y_p ,:) = x;
           loss = 1 - ( w(y,:)* x - w(y_p,:)* x);    %Correct weight argumment is more than 1
           tau = loss/(norm(f_c - f_w)^2); 
           w = w + tau * (f_c - f_w); 
        end 
    end
    
    for iii = 1: length(Test_y)
        x = double(transpose(Test_x(iii,:))); 
        y = Test_y(1,iii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
           count = count+1;
        end 
    end
    
    error_test_2(1,i) = count;
    accuracy_test_2(1,i) = 1 - count/length(Test_y);
end

figure(3)
plot(t,accuracy_test_2,'b--',t,accuracy_test_1,'k--')
grid on
xlim([1 T])
ylim([0.5 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Multi-Class test Accuracy curve')
legend('PA', 'Perceptron')

figure(4)
plot(t,accuracy_test_1,'b--',t,accuracy_training_1,'k--')
grid on
xlim([1 T])
ylim([0.5 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Multi-Class Perceptron test Accuracy curve')
legend('Test', 'Training')

figure(5)
plot(t,accuracy_test_2,'b--',t,accuracy_training_2,'k--')
grid on
xlim([1 T])
ylim([0.5 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Multi-Class PA test Accuracy curve')
legend('Test', 'Training')


%% Multi-class Average Perceptron

w = zeros(k,d);  %intial vector for 784 features
% Training
avg_error1 = zeros(1,T);   % Error count for each T Runs
accuracy_avg1 = zeros(1,T);
u = zeros(k,d); c = zeros(1,2); c1 = 0; 
for i = 1:T
     % average perceptron Training
     for ii = 1: length(Train_y)
         x = double(transpose(Train_x(ii,:)));   %Read training data x
         y = Train_y(1,ii);              
         [a , y_p] = max(w*x);   % Predict training label
         
         if y ~= y_p
            f_c = zeros(k,d);f_c(y ,:) = x;
            f_w = zeros(k,d); f_w(y_p ,:) = x;
            w = w + f_c - f_w;
            count = c(1,2)- c(1,1);
            c(1,1) = c1;
            u = u + count * f_w;
            %u = u + count * f_c(y ,:);
         end
         c1 = c1+1;
      c(1,2) = c1;
     end
     w = w - (1/c1) * u;    
     c2 =0;
     for iii = 1: length(Test_y)
                x = double(transpose(Test_x(iii,:)));   %Read training data x
                y = Test_y(1,iii);
                [a , y_p] = max(w*x);   % Predict training label
         
                if y ~= y_p
                    c2 = c2 + 1;
                end         
     end
     avg_error1(1,i) = c2;
     accuracy_avg1(1,i) = 1 - c2/length(Test_y);
end
 
figure(6)
plot(t,accuracy_avg1,'b--',t,accuracy_test_1,'k--')
grid on
xlim([1 T])
ylim([0.5 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Multi-Class Test Accuracy curve')
legend('Avg Perceptron', 'Perceptron')

figure(7)
plot(t,accuracy_avg1,'b--',t,accuracy_test_2,'k--')
grid on
xlim([1 T])
ylim([0.5 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Multi-Class Test Accuracy curve')
legend('Avg Perceptron', 'PA')

%% Multi-Class Classifications of Test Labels using Perceptron


% used this variable to manually change the number of examples per iteration 
Case = 0;

if Case == 0
    e = [5000 10000 15000 20000 25000 30000  35000 40000 45000 50000 55000 60000];
else 
    e = [35000 40000 45000 50000 55000 60000 5000 10000 15000 20000 25000 30000];
end

% Training
T = 20;    %# of iterations
error_1 = zeros(1,12);   % Error count for each T Runs
accuracy_1 = zeros(1,12);
for l = 1:12
    w = zeros(k,d);  %intial vector for 784 features
    count = 0;
    examples = e(l);        % automating plot generation
    for i = 1:T  
        for j = 1: examples    %Error count during Tth iteration
            x = double(transpose(Train_x(j,:)));   %Read training data x
            y = Train_y(1,j);
            [a , y_p] = max(w*x);
            if y ~= y_p 
               f_c = zeros(k,d); f_c(y ,:) = x;
               f_w = zeros(k,d); f_w(y_p ,:) = x;
               w = w + f_c - f_w; 
            end
        end
    end
    
    for iii = 1: length(Test_y)
        x = double(transpose(Test_x(iii,:))); 
        y = Test_y(1,iii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
            count = count + 1; 
        end
    end
    
    error_1(1,l) = count;
    accuracy_1(1,l) = 1 - count/length(Test_y);
end
    

%% Multi-Class Classifications of Labels using PA

% Training
error_2 = zeros(1,12);   % Error count for each T Runs
accuracy_2 = zeros(1,12);
for l = 1:12
    w = zeros(k,d);  %intial vector for 784 features
    examples = e(l);        % automating plot generation
    count = 0;
    for i = 1:T
     for j = 1: examples    %Error count during Tth iteration
         x = double(transpose(Train_x(j,:)));   %Read training data x
         y = Train_y(1,j);
         [a , y_p] = max(w*x);
         if y ~= y_p 
           f_c = zeros(k,d); f_c(y ,:) = x;
           f_w = zeros(k,d); f_w(y_p ,:) = x;
           loss = 1 - ( w(y,:)* x - w(y_p,:)* x); 
           tau = loss/(norm(f_c - f_w)^2); 
           w = w + tau * (f_c - f_w); 
        end 
     end
    end
    
    for iii = 1: length(Test_y)
        x = double(transpose(Test_x(iii,:))); 
        y = Test_y(1,iii);
        [a , y_p] = max(w*x);
        
        if y ~= y_p 
            count = count + 1; 
        end
    end
    
    error_2(1,l) = count;
    accuracy_2(1,l) = 1 - count/length(Test_y);
end

%% Multi-class Average Perceptron

% Training
error_3 = zeros(1,12);   % Error count for each T Runs
accuracy_3 = zeros(1,12);
for l = 1:12
    w = zeros(k,d);  %intial vector for 784 features
    examples = e(l);        % automating plot generation
    count = 0;
    u = zeros(k,d); c = zeros(1,2); c1 = 0; 
    for i = 1:T
     % average perceptron Training
     for ii = 1: length(Train_y)
         x = double(transpose(Train_x(ii,:)));   %Read training data x
         y = Train_y(1,ii);              
         [a , y_p] = max(w*x);   % Predict training label
         
         if y ~= y_p
            f_c = zeros(k,d);f_c(y ,:) = x;
            f_w = zeros(k,d); f_w(y_p ,:) = x;
            w = w + f_c - f_w;
            count = c(1,2)- c(1,1);
            c(1,1) = c1;
            u = u + count * f_w;
            %u = u + count * f_c(y ,:);
         end
         c1 = c1+1;
      c(1,2) = c1;
     end
     w = w - (1/c1) * u;    
     c2 =0;
     for iii = 1: length(Test_y)
                x = double(transpose(Test_x(iii,:)));   %Read training data x
                y = Test_y(1,iii);
                [a , y_p] = max(w*x);   % Predict training label
         
                if y ~= y_p
                    c2 = c2 + 1;
                end         
     end
    end
    error_3(1,l) = c2;
    accuracy_3(1,l) = 1 - c2/length(Test_y);
end

figure(8)
t = 1:12;
plot(t,accuracy_2,'b--',t,accuracy_1,'k--')
grid on
xlim([1 12])
ylim([0.5 1])
xlabel('Varried Example Sets')
ylabel('Accuracy')
title('Multi-Class General Learning curve')
legend('PA', 'Perceptron')

figure(9)
plot(t,accuracy_3,'b--',t,accuracy_1,'k--')
grid on
xlim([1 12])
ylim([0.5 1])
xlabel('Varried Example Sets')
ylabel('Accuracy')
title('Multi-Class General learning curve')
legend('Avg Perceptron', 'Perceptron')
 

figure(10)
plot(t,accuracy_3,'b--',t,accuracy_2,'k--')
grid on
xlim([1 12])
ylim([0.5 1])
xlabel('Varried Example Sets')
ylabel('Accuracy')
title('Multi-Class General learning curve')
legend('Avg Perceptron', 'PA')

