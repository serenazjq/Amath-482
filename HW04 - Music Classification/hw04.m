clear; close all; clc;
%% 
folderb = dir('Beethoven');
Ab = [];
for i = 4:length(folderb)
    [Y,FS] = audioread(['Beethoven/', folderb(i).name]);
    Ab = [Ab;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        Ab=[Ab;Y(FS*5*(jj -1)+1:FS*5*jj)];
    end
end

folderSG = dir('SG');
ASG = [];
for i = 3:length(folderSG)
    [Y,FS] = audioread(['SG/', folderSG(i).name]);
    ASG = [ASG;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        ASG=[ASG;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

folderMJ = dir('MJ');
AMJ = [];
for i = 4:length(folderMJ)
    [Y,FS] = audioread(['MJ/', folderMJ(i).name]);
    AMJ = [AMJ;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        AMJ=[AMJ;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

n = 124;
Ab=Ab(1:n,:); Ab=Ab(randperm(n),:);
AMJ=AMJ(1:n,:);AMJ=AMJ(randperm(n),:);
ASG=ASG(1:n,:);ASG=ASG(randperm(n),:);
A=[Ab;AMJ;ASG];

spec = [];
for i = 1:size(A,1)
    s = spectrogram(A(i,:));
    spec(i,:) = s(:);
end

[u,s,v] = svd(A - mean(A(:)), 'econ');
%plot(diag(s)/sum(diag(s)),'ro')
%xlabel('Singular Values')
%ylabel('Energey(%)')
%title('Test 1 Singular Value Spectrum(Percentage)')


xtrain=[v(1:62,:);v(125:186,:);v(249:310,:)];
xtest=[v(63:124,:);v(187:248,:);v(311:372,:)];
ytrain=[zeros(62,1)+1;zeros(62,1)+2;zeros(62,1)+3];
ytest=[zeros(62,1)+1;zeros(62,1)+2;zeros(62,1)+3];

classifier = fitcnb(real(xtrain),ytrain);
predicted = predict(classifier, real(xtest));
accuracy1 = sum(predicted==ytest)/length(ytest);

%% test 2 
folderM = dir('Miller');
AM = [];
for i = 4:length(folderM)
    [Y,FS] = audioread(['Miller/', folderM(i).name]);
    AM = [AM;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        AM=[AM;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

folderD = dir('Derek');
AD = [];
for i = 4:length(folderD)
    [Y,FS] = audioread(['Derek/', folderD(i).name]);
    AD = [AD;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        AD=[AD;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

folderT = dir('Thorn');
AT = [];
for i = 4:length(folderT)
    [Y,FS] = audioread(['Thorn/', folderT(i).name]);
    AT = [AT;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        AT=[AT;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

n = 96;
AM=AM(1:n,:); AM=AM(randperm(n),:);
AD=AD(1:n,:);AD=AD(randperm(n),:);
AT=AT(1:n,:);AT=AT(randperm(n),:);
A=[AM;AD;AT];

spec = [];
for i = 1:size(A,1)
    s = spectrogram(A(i,:));
    spec(i,:) = s(:);
end

[u,s,v] = svd(A - mean(A(:)), 'econ');


xtrain=[v(1:16,:);v(33:48,:);v(65:80,:)];
xtest=[v(17:32,:);v(49:64,:);v(81:96,:)];
ytrain=[zeros(16,1)+1;zeros(16,1)+2;zeros(16,1)+3];
ytest=[zeros(16,1)+1;zeros(16,1)+2;zeros(16,1)+3];

classifier = fitcnb(real(xtrain),ytrain);
predicted = predict(classifier, real(xtest));
accuracy2 = sum(predicted==ytest)/length(ytest);

%% TEST 3
folderC = dir('Classical');
AC = [];
for i = 4:length(folderC)
    [Y,FS] = audioread(['Classical/', folderC(i).name]);
    AC = [AC;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        AC=[AC;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

folderCO = dir('Country');
ACO = [];
for i = 3:length(folderCO)
    [Y,FS] = audioread(['Country/', folderCO(i).name]);
    ACO = [ACO;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        ACO=[ACO;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

folderH = dir('Hiphop');
AH = [];
for i = 3:length(folderH)
    [Y,FS] = audioread(['Hiphop/', folderH(i).name]);
    AH = [AH;Y(1:FS*5)];
    for jj=2:length(Y)/FS/5-1
        AH=[AH;Y(FS*5*(jj-1)+1:FS*5*jj)];
    end
end

n = 240;
AC=AC(1:n,:); AC=AC(randperm(n),:);
ACO=ACO(1:n,:);ACO=ACO(randperm(n),:);
AH=AH(1:n,:);AH=AH(randperm(n),:);
A=[AC;ACO;AH];

spec = [];
for i = 1:size(A,1)
    s = spectrogram(A(i,:));
    spec(i,:) = s(:);
end

[u,s,v] = svd(A - mean(A(:)), 'econ');


xtrain=[v(1:40,:);v(81:120,:);v(161:200,:)];
xtest=[v(41:80,:);v(121:160,:);v(201:240,:)];
ytrain=[zeros(40,1)+1;zeros(40,1)+2;zeros(40,1)+3];
ytest=[zeros(40,1)+1;zeros(40,1)+2;zeros(40,1)+3];

classifier = fitcnb(real(xtrain),ytrain);
predicted = predict(classifier, real(xtest));
accuracy3 = sum(predicted==ytest)/length(ytest);