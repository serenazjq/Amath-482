clear; close all; clc; 
%% test 1
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');

[xlength1, ylength1,a1,num_f1] = size(vidFrames1_1);
data1 = zeros(2,num_f1);
for i = 1:num_f1
    img = rgb2gray(vidFrames1_1(:,:,:,i));
    X1 = double(img(:,320:380));
    [V,I] = max(X1(:));
    [newy,newx] = ind2sub(size(X1),I);
    data1(:,i) = [newx;newy];
end


[xlength2,ylength2,a2,num_f2] = size(vidFrames2_1);
data2 = zeros(2,num_f2);
for i = 1:num_f2
    img = rgb2gray(vidFrames2_1(:,:,:,i));
    X2 = double(img(:,260:340));
    [V,I] = max(X2(:));
    [newy,newx] = ind2sub(size(X2),I);
    data2(:,i) = [newx;newy];
end

[xlength3,ylength3,a3,num_f3] = size(vidFrames3_1);
data3 = zeros(2,num_f3);
for i = 1:num_f3
    img = rgb2gray(vidFrames3_1(:,:,:,i));
    X3 = double(img(250:310,290:360));
    [V,I] = max(X3(:));
    [newy,newx] = ind2sub(size(X3),I);
    data3(:,i) = [newx;newy];
end

minLength = 226;
data1 = data1(:,1:minLength);
data2 = data2(:,10:minLength+9);
data3 = data3(:,1:minLength);

figure(1)
subplot(2,1,1);
plot(data1(1,:)-mean(data1(1,:))), hold on
plot(data2(1,:)-mean(data2(1,:)))
plot(data3(2,:)-mean(data3(2,:)))
xlabel('Time')
ylabel('Displacement in x direction')
ylim([-100, 200])
title('Test 1 in x direction')
legend('cam1','cam2','cam3');
subplot(2,1,2);
plot(data1(2,:)-mean(data1(2,:))), hold on
plot(data2(2,:)-mean(data2(2,:)))
plot(data3(1,:)-mean(data3(1,:)))
xlabel('Time')
ylabel('Displacement in y direction')
ylim([-150, 150])
title('Test 1 in y direction')
legend('cam1','cam2','cam3');
hold off


X = [data1;data2;data3];
[m,n] = size(X);
mn = mean(X,2);
X = X - repmat(mn,1,n);
[u,s,v]=svd(X'/sqrt(n-1));
lambda=diag(s).^2;
figure(2)
plot(lambda/sum(lambda),'o');
xlabel('Mode')
ylabel('Principle Component')
title('Test 1 Principle Component Analysis')

%% test 2
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');

[xlength1, ylength1,a1,num_f1] = size(vidFrames1_2);
data1 = zeros(2,num_f1);
for i = 1:num_f1
    img = rgb2gray(vidFrames1_2(:,:,:,i));
    X1 = double(img(200:400,300:400));
    [V,I] = max(X1(:));
    [newy,newx] = ind2sub(size(X1),I);
    data1(:,i) = [newx;newy];
end


[xlength2,ylength2,a2,num_f2] = size(vidFrames2_2);
data2 = zeros(2,num_f2);
for i = 1:num_f2
    img = rgb2gray(vidFrames2_2(:,:,:,i));
    X2 = double(img(50:370,200:400));
    [V,I] = max(X2(:));
    [newy,newx] = ind2sub(size(X2),I);
    data2(:,i) = [newx;newy];
end

[xlength3,ylength3,a3,num_f3] = size(vidFrames3_2);
data3 = zeros(2,num_f3);
for i = 1:num_f3
    img = rgb2gray(vidFrames3_2(:,:,:,i));
    X3 = double(img(180:320,250:460));
    [V,I] = max(X3(:));
    [newy,newx] = ind2sub(size(X3),I);
    data3(:,i) = [newx;newy];
end

minLength = 314;
data1 = data1(:,1:minLength);
data2 = data2(:,20:minLength+19);
data3 = data3(:,1:minLength);

figure(1)
subplot(2,1,1);
plot(data1(1,:)-mean(data1(1,:))), hold on
plot(data2(1,:)-mean(data2(1,:)))
plot(data3(2,:)-mean(data3(2,:)))
xlabel('Time')
ylabel('Displacement in x direction')
ylim([-100, 100])
title('Test 2 in x direction')
legend('cam1','cam2','cam3');
subplot(2,1,2);
plot(data1(2,:)-mean(data1(2,:))), hold on
plot(data2(2,:)-mean(data2(2,:)))
plot(data3(1,:)-mean(data3(1,:)))
xlabel('Time')
ylabel('Displacement in y direction')
ylim([-150, 150])
title('Test 2 in y direction')
legend('cam1','cam2','cam3');
hold off


X = [data1;data2;data3];
[m,n] = size(X);
mn = mean(X,2);
X = X - repmat(mn,1,n);
[u,s,v]=svd(X'/sqrt(n-1));
lambda=diag(s).^2;
figure(2)
plot(lambda/sum(lambda),'o');
xlabel('Mode')
ylabel('Principle Component')
title('Test 2 Principle Component Analysis')

%% test 3
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');

[xlength1, ylength1,a1,num_f1] = size(vidFrames1_3);
data1 = zeros(2,num_f1);
for i = 1:num_f1
    img = rgb2gray(vidFrames1_3(:,:,:,i));
    X1 = double(img(250:400,200:400));
    [V,I] = max(X1(:));
    [newy,newx] = ind2sub(size(X1),I);
    data1(:,i) = [newx;newy];
end


[xlength2,ylength2,a2,num_f2] = size(vidFrames2_3);
data2 = zeros(2,num_f2);


for i = 1:num_f2
    img = rgb2gray(vidFrames2_3(:,:,:,i));
    X2 = double(img(200:400,220:380));
    [V,I] = max(X2(:));
    [newy,newx] = ind2sub(size(X2),I);
    data2(:,i) = [newx;newy];
end

[xlength3,ylength3,a3,num_f3] = size(vidFrames3_3);
data3 = zeros(2,num_f3);
for i = 1:num_f3
    img = rgb2gray(vidFrames3_3(:,:,:,i));
    X3 = double(img(180:350,150:480));
    [V,I] = max(X3(:));
    [newy,newx] = ind2sub(size(X3),I);
    data3(:,i) = [newx;newy];
end

minLength = 237;
data1 = data1(:,1:minLength);
data2 = data2(:,30:minLength+29);
data3 = data3(:,1:minLength);

figure(1)
subplot(2,1,1);
plot(data1(1,:)-mean(data1(1,:))), hold on
plot(data2(1,:)-mean(data2(1,:)))
plot(data3(2,:)-mean(data3(2,:)))
xlabel('Time')
ylabel('Displacement in x direction')
ylim([-100, 100])
title('Test 3 in x direction')
legend('cam1','cam2','cam3');
subplot(2,1,2);
plot(data1(2,:)-mean(data1(2,:))), hold on
plot(data2(2,:)-mean(data2(2,:)))
plot(data3(1,:)-mean(data3(1,:)))
xlabel('Time')
ylabel('Displacement in y direction')
ylim([-150, 150])
title('Test 3 in y direction')
legend('cam1','cam2','cam3');
hold off


X = [data1;data2;data3];
[m,n] = size(X);
mn = mean(X,2);
X = X - repmat(mn,1,n);
[u,s,v]=svd(X'/sqrt(n-1));
lambda=diag(s).^2;
figure(2)
plot(lambda/sum(lambda),'o');
xlabel('Mode')
ylabel('Principle Component')
title('Test 3 Principle Component Analysis')

%% test 4
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');

[xlength1, ylength1,a1,num_f1] = size(vidFrames1_4);
data1 = zeros(2,num_f1);
for i = 1:num_f1
    img = rgb2gray(vidFrames1_4(:,:,:,i));
    X1 = double(img(200:380,300:480));
    [V,I] = max(X1(:));
    [newy,newx] = ind2sub(size(X1),I);
    data1(:,i) = [newx;newy];
end


[xlength2,ylength2,a2,num_f2] = size(vidFrames2_4);
data2 = zeros(2,num_f2);


for i = 1:num_f2
    img = rgb2gray(vidFrames2_4(:,:,:,i));
    X2 = double(img(80:400,220:420));
    [V,I] = max(X2(:));
    [newy,newx] = ind2sub(size(X2),I);
    data2(:,i) = [newx;newy];
end

[xlength3,ylength3,a3,num_f3] = size(vidFrames3_4);
data3 = zeros(2,num_f3);
for i = 1:num_f3
    img = rgb2gray(vidFrames3_4(:,:,:,i));
    X3 = double(img(150:300,300:520));
    [V,I] = max(X3(:));
    [newy,newx] = ind2sub(size(X3),I);
    data3(:,i) = [newx;newy];
end

minLength = 392;
data1 = data1(:,1:minLength);
data2 = data2(:,1:minLength);
data3 = data3(:,1:minLength);


figure(1)
subplot(2,1,1);
plot(data1(1,:)-mean(data1(1,:))), hold on
plot(data2(1,:)-mean(data2(1,:)))
plot(data3(2,:)-mean(data3(2,:)))
xlabel('Time')
ylabel('Displacement in x direction')
ylim([-100, 100])
title('Test 3 in x direction')
legend('cam1','cam2','cam3');
subplot(2,1,2);
plot(data1(2,:)-mean(data1(2,:))), hold on
plot(data2(2,:)-mean(data2(2,:)))
plot(data3(1,:)-mean(data3(1,:)))
xlabel('Time')
ylabel('Displacement in y direction')
ylim([-150, 150])
title('Test 4 in y direction')
legend('cam1','cam2','cam3');
hold off


X = [data1;data2;data3];
[m,n] = size(X);
mn = mean(X,2);
X = X - repmat(mn,1,n);
[u,s,v]=svd(X'/sqrt(n-1));
lambda=diag(s).^2;
figure(2)
plot(lambda/sum(lambda),'o');
xlabel('Mode')
ylabel('Principle Component')
title('Test 4 Principle Component Analysis')

   