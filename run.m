clear;
%clc;
warning off;
addpath(genpath('./'));

DBDIR = './Dataset/';
%DBDIR = 'D:/Data/';
%% dataset
i= 1;
DataName{i} = 'Caltech101-20'; i = i + 1;
dbNum = length(DataName);

for dsi = 1:dbNum
    clear X gt Y;
    dataName = DataName{dsi}; 
    dbfilename = sprintf('%s%s.mat',DBDIR,dataName);
    load(dbfilename);
    Y= gt;    
    k = length(unique(Y));  
    
    %% para setting
   
    anchor = [1] * k;
    
%	d = [1,2,4]*k ;   
%	alpha = [0.01, 0.1, 0.5,  1,  2, 10];    
%	beta = [0.0001,0.001, 0.01,0.1,1];   
%	lambda = [0.01，0.1, 1, 10];
   alpha = 1;
   d = 2*k;
   beta = 0.01;
   lambda = 1;
   %if the result is not good, please modify the seed of rng function in  the REIEBO.m  or modify the parameters
   if contains(dataName,'Caltech101-20')       % m:20  Dim:40  beta:0.0100 lambda:10.0000  alpha:0.0100     
        alpha = 0.01; 
        beta = 0.01; 
        lambda = 10;
   %if the result is not good, please modify the seed of rng function in  the REIEBO.m  or modify the parameters
   elseif contains(dataName,'YouTubeFace50_4Views')    %c:50 m:50  Dim:100  beta:0.1000 lambda:0.0100  alpha:0.0100  
        lambda = 0.01; 
        beta = 0.1;
        alpha = 0.01;    
    end 

    
    tic;                       
    [F, iter] = REIEBO(X,Y,d,anchor, alpha,beta, lambda); 
    
    [~,idx]=max(F);                   
    res = Clustering8Measure(Y, idx); 

    str = sprintf('db:%s\t  m:%d Anchor:%d\t Dim:%d\t \t beta:%.4f\t alpha:%.4f\t ACC:%.4f nmi:%.4f AR:%.4f Fscore:%.4f Purity:%.4f  Precision:%.4f Recall:%.4f    \tTime:%.4f %s\n',...
        dataName, k, anchor, d, beta, alpha, res(1), res(2), res(3), res(4), res(5), res(6), res(7), toc, GetTimeStrForLog());
    fprintf(str);

    clear F X Y k

end


