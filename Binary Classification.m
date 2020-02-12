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
Train_x = normalize(Train_x);
%10,000 Test data available
Test_x = zeros(length(data_1),784); % Creating labels for Binary classification training
for i = 1: length(data_1)
    for j = 2:785
        Test_x(i,j-1) = data_1(i,j);   %Read training label y
    end
end

Test_x = normalize(Test_x);


%60,000 Training labels available
Train_y = zeros(1,length(data)); % Creating labels for Binary classification training
%Train_x(i) = double(Tain_Images(i,:)) Features during the ith iteration in
%only double class operations
for i = 1: length(data)
        y = int8(data(i,1));   %Read training label y
        
        if rem(y,2) == 0         % assign Training labeln
            Train_y(1,i) = 1;      % even  class 1
        else
            Train_y(1,i) = -1;     % odd class -1
        end
end

%10,000 test labels available
Test_y = zeros(1,length(data_1)); % Creating labels for Binary classification test
for i = 1: length(data_1)
        y = int8(data_1(i,1));   %Read training data x
        
        if rem(y,2) == 0         % assign Training label
            Test_y(1,i) = 1;      % even  class 1
        else
            Test_y(1,i) = -1;       % odd class -1
        end
end

%% Q5.1 _ a Online Learning Curve

%Binary Classifications of Labels using Perceptron
% classify the labels into odd or even
% Even class label = 1, odd class label = -1 

% Training

T = 50;    %# of iterations
w = zeros(1,784);  %intial vector for 784 features
error_1 = zeros(1,T);   % Error count for each T Runs
weight = zeros(1,784); %  weights after Tth iteration
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for j = 1: length(Train_y)
        x = double(Train_x(j,:));   %Read training data x
        y = Train_y(1,j);
        
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
            w = w +  y * x;
            count = count+1;
        end
    end
    error_1(1,i) = count;
    weight = [weight ; w];
end


% Binary Classifications of Labels using PA
% classify the labels into odd or even
% Even class label = 1, odd class label = -1 
% 
% Training

w = zeros(1,784);  %intial vector for 784 features
error_2 = zeros(1,T);   % Error count for each T Runs
weight_pa = zeros(1,784); %  weights after Tth iteration
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for j = 1: length(Train_y)
        x = double(Train_x(j,:));   %Read training data x
        y = Train_y(1,j);
        
        y_p = sign( dot(x, w));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
            count = count+1;
        end
        loss = max(0,0.5 - y * dot(x, w));
        tau = loss/(norm(x)^2) ;
        w = w +  tau * y * x;
    end
    error_2(1,i) = count;
    weight_pa = [weight_pa ; w];
end


figure(1)
t = 1:50;
plot(t,error_1,'b--',t,error_2,'r--')
grid on
xlim([1 51])
xlabel('Iterations')
ylim([0 max(error_1)+5])
ylabel('Mistakes')
title('Online Learning Curve comparison')
legend('Perceptron', 'PA algo')


%% Q5.1_b_Training curve

%Perceptron accuracy on Training data

T = 20;    %# of iterations
w = zeros(1,784);   %intial vector for 784 features
error1_accuracy_training = zeros(1,T);   % Error count for each T Runs
weight1_accuracy_training = zeros(T,784); %  weights after Tth iteration
training_accuracy1 = zeros(1,T);
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for j = 1: length(Train_y)
        x = double(Train_x(j,:));   %Read training data x
        y = Train_y(1,j);
        
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
            w = w +  y * x;
            count = count+1;
        end
    end
    error1_accuracy_training(1,i) = count;                      %#of correct pedictions/total predictions
    weight1_accuracy_training(i,:) = w;
    training_accuracy1(1,i) = 1 - count/length(Train_y);    % during the T-th iteration pass
end

%PA algo Online Curve

w = zeros(1,784);   %intial vector for 784 features
error2_accuracy_training = zeros(1,T);   % Error count for each T Runs
weight2_accuracy_training = zeros(T,784); %  weights after Tth iteration
training_accuracy2 = zeros(1,T);
for i = 1:T
    count = 0;        %Error count during Tth iteration
    for j = 1: length(Train_y)
        x = double(Train_x(j,:));   %Read training data x
        y = Train_y(1,j);
        
        y_p = sign( dot(x, w));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
            count = count+1;
        end
        loss = max(0,1 - y * dot(x, w));
        tau = loss/(norm(x)^2) ;
        w = w +  tau * y * x;
    end
    error2_accuracy_training(1,i) = count;
    weight2_accuracy_training(i,:) = w;
    training_accuracy2(1,i) = 1 - count/length(Train_y);
end

figure(2)
t = 1:20;
plot(t,training_accuracy2,'b--',t,training_accuracy1,'r--')
grid on
xlim([1 21])
ylim([0.7 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Accuracy curve on Training set')
legend('PA','Perceptron')

%% Q5.1_b_Test curve
%chose a weight vector from training set after the Tth iteration
%Checking performance on this Tth weight
%Perceptron accuracy on Test data

T = 20;    %# of iterations
error1_accuracy_test = zeros(1,T);   % Error count for each T Runs
test_accuracy1 = zeros(1,T);
for i = 1:T
    w = weight1_accuracy_training(i,:);   %intial vector for 784 features
    count = 0;        %Error count during Tth iteration
    for j = 1: length(Test_y)
        x = double(Test_x(j,:));   %Read training data x
        y = Test_y(1,j);
        
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
            count = count+1;
        end
    end
    error1_accuracy_test(1,i) = count;                      %#of correct pedictions/total predictions
    test_accuracy1(1,i) = 1 - count/length(Test_y);    % during the T-th iteration pass
end

%PA test accuracy using Tth iteration weight

error2_accuracy_test = zeros(1,T);   % Error count for each T Runs
test_accuracy2 = zeros(1,T);
for i = 1:T
    w = weight2_accuracy_training(i,:);   %intial vector for 784 features
    count = 0;        %Error count during Tth iteration
    for j = 1: length(Test_y)
        x = double(Test_x(j,:));   %Read training data x
        y = Test_y(1,j);
        
        y_p = sign( dot(x, w));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
            count = count+1;
        end
    end
    error2_accuracy_test(1,i) = count;
    test_accuracy2(1,i) = 1 - count/length(Test_y);
end
figure(3)
t = 1:20;
plot(t,test_accuracy2,'b--',t,test_accuracy1,'r--')
grid on
xlim([1 21])
ylim([0.7 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Accuracy curve on Test set')
legend('PA','Perceptron')

figure(4)
t = 1:20;
plot(t,training_accuracy1,'b--',t,test_accuracy1,'r--')
grid on
xlim([1 21])
ylim([0.7 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Perceptron Accuracy curve')
legend('Training','Test')

figure(5)
t = 1:20;
plot(t,training_accuracy2,'b--',t,test_accuracy2,'r--')
grid on
xlim([1 21])
ylim([0.7 1])
xlabel('Iterations')
ylabel('Accuracy')
title('PA Accuracy curve')
legend('Training','Test')

%% Q5_c Average Perceptron

% Binary Classifications of Labels using Average Perceptron
% classify the labels into odd or even
% Even class label = 1, odd class label = -1 

% Training
w = zeros(1,784); %  weights after Tth iteration
weight = zeros(T,784);
u = zeros(1,784); 
count = 1;  
for i = 1:T
    c = 0;
    % average perceptron Training
    for ii = 1: length(Train_y)
        x = double(Train_x(ii,:));   %Read training data x
        y = Train_y(1,ii);              
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
           w = w +  y * x;
           u = u + y * count * x;
           c = c + 1;
        end
     count = count+1;
    end
    weight(i,:) = w - (1/count) * u; 
end

    
 % Test Accuracy

avg_error2 = zeros(1,T);
accuracy_avg2 = zeros(1,T);
 for i = 1:T
     w = weight(i,:);
     c = 0;
     for ii = 1: length(Test_y)
        x = double(Test_x(ii,:));   %Read training data x
        y = Test_y(1,ii);              
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
        if y ~= y_p
           c = c + 1;
        end
     end
    avg_error2(1,i) = c;
    accuracy_avg2(1,i) = 1 - c/length(Test_y);
 end


figure(6)
plot(t,accuracy_avg2,'b--',t,test_accuracy1,'r--')
grid on
xlim([1 21])
ylim([0.7 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Accuracy curve on Test set')
legend('Average Perceptron','Perceptron Accuracy')

figure(7)
plot(t,accuracy_avg2,'b--',t,test_accuracy2,'r--')
grid on
xlim([1 21])
ylim([0.7 1])
xlabel('Iterations')
ylabel('Accuracy')
title('Accuracy curve on Test set')
legend('Average Perceptron','PA')


%% General Learning Curve
% classify the labels into odd or even
% Even class label = 1, odd class label = -1 

% used this variable to manually change the number of examples per iteration 

Case = 1;    % 0 or 1

if Case == 0
    k = [5000 10000 15000 20000 25000 30000  35000 40000 45000 50000 55000 60000];
else 
    k = [35000 40000 45000 50000 55000 60000 5000 10000 15000 20000 25000 30000];
end

% Training
T = 20;    %# of iterations
error_1 = zeros(1,12);   % Error count for each T Runs
accuracy_1 = zeros(1,12);
weight = zeros(12,784);
for l = 1:12
    w = zeros(1,784);  %intial vector for 784 features
    examples = k(l);        % automating plot generation
    for i = 1:T
        
     %Training weight vector   
     for j = 1: examples    %Error count during Tth iteration
         x = double(Train_x(j,:));   %Read training data x
         y = Train_y(1,j);
        
         y_p = sign( dot(w , x));   % Predict training label
         if y_p == 0
             y_p = 1;               %Matlab sign returns 0
         end
        
         if y ~= y_p
             w = w +  y * x;
         end 
     end 
    end
    weight(l,:) = w;
end

for l = 1:12
    count = 0;
    w = weight(l,:);
    for ii = 1 : length(Test_y)
        x = double(Test_x(ii,:));   %Read training data x
        y = Test_y(1,ii);
        
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
         if y ~= y_p
             %w = w +  y * x;
             count = count+1;
         end 
    end
   error_1(1,l) = count;
   accuracy_1(1,l) = 1 - count/length(Test_y);
end

%% Binary Classifications of Labels using PA
% classify the labels into odd or even
% Even class label = 1, odd class label = -1 

% Training
error_2 = zeros(1,12);   % Error count for each T Runs
accuracy_2 = zeros(1,12);
weight_pa = zeros(12,784);
for l = 1:12
    w = zeros(1,784);  %intial vector for 784 features
    examples = k(l);        % automating plot generation
    count = 0;
    for i = 1:T
     for j = 1: examples    %Error count during Tth iteration
         x = double(Train_x(j,:));   %Read training data x
         y = Train_y(1,j);
        
         y_p = sign( dot(w , x));   % Predict training label
         if y_p == 0
             y_p = 1;               %Matlab sign returns 0
         end
        
        if y ~= y_p
            count = count+1;
        end
        loss = max(0,1 - y * dot(x, w));
        tau = loss/(norm(x)^2) ;
        w = w +  tau * y * x;
     end
    end
    weight_pa(l,:) = w;
end

for l = 1:12
    count = 0;
    w = weight_pa(l,:);
    for ii = 1 : length(Test_y)
        x = double(Test_x(ii,:));   %Read training data x
        y = Test_y(1,ii);
        
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
         if y ~= y_p
             count = count+1;
         end 
    end
   error_2(1,l) = count;
   accuracy_2(1,l) = 1 - count/length(Test_y);
end


%% Average Perceptron


% Training
weight_avg = zeros(12,784);

% Training
for l = 1:12
    w = zeros(1,784);  %intial vector for 784 features
    examples = k(l);        % automating plot generation
    count = 1;
    u = zeros(1,784);
    for i = 1:T
    % average perceptron Training
     for j = 1: examples    %Error count during Tth iteration
         x = double(Train_x(j,:));   %Read training data x
         y = Train_y(1,j);            
         y_p = sign( dot(w , x));   % Predict training label
         if y_p == 0
            y_p = 1;               %Matlab sign returns 0
         end
        
         if y ~= y_p
              w = w +  y * x;
              u = u + y * count * x;
         end
      count = count+1;
     end 
    end
    weight_avg(l,:) = w - (1/count) * u;
end


    
 % Test Accuracy

avg_error2 = zeros(1,12);
accuracy_avg2 = zeros(1,12);
for l = 1:12
    c = 0;
    w = weight_avg(l,:);
    for ii = 1 : length(Test_y)
        x = double(Test_x(ii,:));   %Read training data x
        y = Test_y(1,ii);
        
        y_p = sign( dot(w , x));   % Predict training label
        if y_p == 0
            y_p = 1;               %Matlab sign returns 0
        end
        
         if y ~= y_p
             c = c+1;
         end 
    end
    avg_error2(1,l) = c;
    accuracy_avg2(1,l) = 1 - c/length(Test_y);
end

figure(8)
t = 1:12;
plot(t,accuracy_2,'b--',t,accuracy_1,'r--')
grid on
xlim([1 12])
ylim([0.7 1])
xlabel('Varried Example Sets')
ylabel('Accuracy')
title('General learning curve for PA and Perceptron')
legend('PA', 'Perceptron') 

figure(9)
plot(t,accuracy_avg2,'b--',t,accuracy_1,'k--')
grid on
xlim([1 12])
ylim([0.7 1])
xlabel('Varried Example Sets')
ylabel('Accuracy')
title('General learning curve comparison')
legend('Avg Perceptron','Perceptron')

figure(10)
plot(t,accuracy_avg2,'b--',t,accuracy_2,'k--')
grid on
xlim([1 12])
ylim([0.7 1])
xlabel('Varried Example Sets')
ylabel('Accuracy')
title('General learning curve comparison')
legend('Avg Perceptron','PA')
 