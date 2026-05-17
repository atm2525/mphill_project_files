clc;
clear;
close all;

%% =========================================================
% PV POWER PREDICTION USING:
% 1. Standard ANN
% 2. FA-ANN
% 3. RAC-FA-ANN
%
% Dataset: PV_Data.csv
% ==========================================================

%% Load Dataset

data = readtable('PV_Data.csv','VariableNamingRule','preserve');

disp('Available columns in dataset:');
disp(data.Properties.VariableNames');

%% Extract Inputs and Output

X = [
    data.("MODULE_TEMP")'
    data.("Amb_Temp")'
    data.("WIND_Speed")'
    data.("IRR (W/m2)")'
    data.("DC Current in Amps")'
    data.("AC Ir in Amps")'
    data.("AC Iy in Amps")'
    data.("AC Ib in Amps")'
];

Y = data.("AC Power in Watts")';

%% Remove Missing Values

valid = all(~isnan(X),1) & ~isnan(Y);

X = X(:,valid);
Y = Y(:,valid);

%% Optional Sampling for Faster Optimization

sampleSize = min(3000, size(X,2));

rng(1);
idx = randperm(size(X,2), sampleSize);

X = X(:,idx);
Y = Y(:,idx);

%% Normalize Data

[Xn, psX] = mapminmax(X,0,1);
[Yn, psY] = mapminmax(Y,0,1);

%% Train / Validation / Test Split

N = size(Xn,2);

rng(1);
indices = randperm(N);

trainRatio = 0.70;
valRatio = 0.15;

nTrain = round(trainRatio*N);
nVal = round(valRatio*N);

trainInd = indices(1:nTrain);
valInd = indices(nTrain+1:nTrain+nVal);
testInd = indices(nTrain+nVal+1:end);

Xtrain = Xn(:,trainInd);
Ytrain = Yn(:,trainInd);

Xval = Xn(:,valInd);
Yval = Yn(:,valInd);

Xtest = Xn(:,testInd);
Ytest = Yn(:,testInd);

%% ANN Structure

inputSize = size(Xtrain,1);
hiddenSize = 20;
outputSize = 1;

numWeights = inputSize*hiddenSize + hiddenSize + hiddenSize*outputSize + outputSize;

fprintf('\nNumber of Inputs = %d\n', inputSize);
fprintf('Hidden Neurons = %d\n', hiddenSize);
fprintf('Total ANN Parameters to Optimize = %d\n', numWeights);

%% =========================================================
% 1. STANDARD ANN
% ==========================================================

fprintf('\nTraining Standard ANN...\n');

net = feedforwardnet(hiddenSize);

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

net.trainFcn = 'trainlm';

net = train(net, Xn, Yn);

Ypred_ANN_n = net(Xtest);

Ypred_ANN = mapminmax('reverse', Ypred_ANN_n, psY);
Yactual = mapminmax('reverse', Ytest, psY);

[RMSE_ANN, MAE_ANN, MAPE_ANN, R2_ANN, R_ANN] = PerformanceMetrics(Yactual,Ypred_ANN);

%% =========================================================
% 2. FA-ANN
% ==========================================================

fprintf('\nTraining ANN using Standard FA...\n');

SearchAgents = 30;
MaxIter = 100;

lb = -1;
ub = 1;
dim = numWeights;

fobj_FA = @(w) ANN_Fitness(w,inputSize,hiddenSize,outputSize,Xtrain,Ytrain,Xval,Yval);

[Best_FA, Curve_FA, BestWeights_FA] = StandardFA(SearchAgents,MaxIter,lb,ub,dim,fobj_FA);

Ypred_FA_n = ANN_Predict(BestWeights_FA,inputSize,hiddenSize,outputSize,Xtest);

Ypred_FA = mapminmax('reverse',Ypred_FA_n,psY);

[RMSE_FA, MAE_FA, MAPE_FA, R2_FA, R_FA] = PerformanceMetrics(Yactual,Ypred_FA);

%% =========================================================
% 3. RAC-FA-ANN
% ==========================================================

fprintf('\nTraining ANN using RAC-FA...\n');

fobj_RAC = @(w) ANN_Fitness(w,inputSize,hiddenSize,outputSize,Xtrain,Ytrain,Xval,Yval);

[Best_RAC, BestWeights_RAC, Curve_RAC] = RAC_FA(SearchAgents,MaxIter,lb,ub,dim,fobj_RAC,[]);

Ypred_RAC_n = ANN_Predict(BestWeights_RAC,inputSize,hiddenSize,outputSize,Xtest);

Ypred_RAC = mapminmax('reverse',Ypred_RAC_n,psY);

[RMSE_RAC, MAE_RAC, MAPE_RAC, R2_RAC, R_RAC] = PerformanceMetrics(Yactual,Ypred_RAC);

%% =========================================================
% Results Table
% ==========================================================

Model = {'Standard ANN'; 'FA-ANN'; 'RAC-FA-ANN'};

RMSE = [RMSE_ANN; RMSE_FA; RMSE_RAC];
MAE = [MAE_ANN; MAE_FA; MAE_RAC];
MAPE = [MAPE_ANN; MAPE_FA; MAPE_RAC];
R = [R_ANN; R_FA; R_RAC];
R2 = [R2_ANN; R2_FA; R2_RAC];

Results = table(Model,RMSE,MAE,MAPE,R,R2);

disp(' ');
disp('================ PV POWER PREDICTION RESULTS ================');
disp(Results);

%% =========================================================
% Figure 1: Actual PV Power Only
% ==========================================================

figure('Name','Actual PV Power','NumberTitle','off');

plot(Yactual,'k','LineWidth',1.5);

grid on;
xlabel('Sample');
ylabel('Actual AC Power in Watts');
title('Actual PV Power');

%% =========================================================
% Figure 2: Predicted PV Power Only
% ==========================================================

figure('Name','Predicted PV Power','NumberTitle','off');

plot(Ypred_ANN,'r','LineWidth',1.2);
hold on;
plot(Ypred_FA,'b','LineWidth',1.2);
plot(Ypred_RAC,'g','LineWidth',1.2);

grid on;
xlabel('Sample');
ylabel('Predicted AC Power in Watts');
title('Predicted PV Power');
legend('ANN','FA-ANN','RAC-FA-ANN','Location','best');

%% =========================================================
% Figure 3: Actual and Predicted Comparison in Separate Subplots
% ==========================================================

figure('Name','Actual vs Predicted Comparison','NumberTitle','off');

subplot(3,1,1)
plot(Yactual,'k','LineWidth',1.3);
hold on;
plot(Ypred_ANN,'r','LineWidth',1.1);
grid on;
xlabel('Sample');
ylabel('Power (W)');
title(sprintf('Standard ANN: RMSE = %.4f, MAE = %.4f, MAPE = %.4f%%, R = %.5f, R^2 = %.5f', ...
    RMSE_ANN, MAE_ANN, MAPE_ANN, R_ANN, R2_ANN));
legend('Actual','ANN Prediction','Location','best');

subplot(3,1,2)
plot(Yactual,'k','LineWidth',1.3);
hold on;
plot(Ypred_FA,'b','LineWidth',1.1);
grid on;
xlabel('Sample');
ylabel('Power (W)');
title(sprintf('FA-ANN: RMSE = %.4f, MAE = %.4f, MAPE = %.4f%%, R = %.5f, R^2 = %.5f', ...
    RMSE_FA, MAE_FA, MAPE_FA, R_FA, R2_FA));
legend('Actual','FA-ANN Prediction','Location','best');

subplot(3,1,3)
plot(Yactual,'k','LineWidth',1.3);
hold on;
plot(Ypred_RAC,'g','LineWidth',1.1);
grid on;
xlabel('Sample');
ylabel('Power (W)');
title(sprintf('RAC-FA-ANN: RMSE = %.4f, MAE = %.4f, MAPE = %.4f%%, R = %.5f, R^2 = %.5f', ...
    RMSE_RAC, MAE_RAC, MAPE_RAC, R_RAC, R2_RAC));
legend('Actual','RAC-FA-ANN Prediction','Location','best');

%% =========================================================
% Figure 4: Regression / Correlation Plots
% ==========================================================

figure('Name','Regression Plots','NumberTitle','off');

subplot(1,3,1)
scatter(Yactual,Ypred_ANN,15,'filled');
hold on;
plot([min(Yactual) max(Yactual)],[min(Yactual) max(Yactual)],'k--','LineWidth',1.5);
grid on;
xlabel('Actual Power (W)');
ylabel('Predicted Power (W)');
title(sprintf('ANN Regression\nR = %.5f, R^2 = %.5f', R_ANN, R2_ANN));

subplot(1,3,2)
scatter(Yactual,Ypred_FA,15,'filled');
hold on;
plot([min(Yactual) max(Yactual)],[min(Yactual) max(Yactual)],'k--','LineWidth',1.5);
grid on;
xlabel('Actual Power (W)');
ylabel('Predicted Power (W)');
title(sprintf('FA-ANN Regression\nR = %.5f, R^2 = %.5f', R_FA, R2_FA));

subplot(1,3,3)
scatter(Yactual,Ypred_RAC,15,'filled');
hold on;
plot([min(Yactual) max(Yactual)],[min(Yactual) max(Yactual)],'k--','LineWidth',1.5);
grid on;
xlabel('Actual Power (W)');
ylabel('Predicted Power (W)');
title(sprintf('RAC-FA-ANN Regression\nR = %.5f, R^2 = %.5f', R_RAC, R2_RAC));

%% =========================================================
% Figure 5: Prediction Error Plot
% ==========================================================

figure('Name','Prediction Error','NumberTitle','off');

plot(Yactual - Ypred_ANN,'r','LineWidth',1.2);
hold on;
plot(Yactual - Ypred_FA,'b','LineWidth',1.2);
plot(Yactual - Ypred_RAC,'g','LineWidth',1.2);

grid on;
xlabel('Sample');
ylabel('Prediction Error');
title('Prediction Error Comparison');
legend('ANN Error','FA-ANN Error','RAC-FA-ANN Error','Location','best');

%% =========================================================
% Figure 6: Absolute Error Plot
% ==========================================================

figure('Name','Absolute Prediction Error','NumberTitle','off');

plot(abs(Yactual - Ypred_ANN),'r','LineWidth',1.2);
hold on;
plot(abs(Yactual - Ypred_FA),'b','LineWidth',1.2);
plot(abs(Yactual - Ypred_RAC),'g','LineWidth',1.2);

grid on;
xlabel('Sample');
ylabel('Absolute Error');
title('Absolute Prediction Error Comparison');
legend('ANN','FA-ANN','RAC-FA-ANN','Location','best');

%% =========================================================
% Figure 7: FA-ANN vs RAC-FA-ANN Convergence
% ==========================================================

figure('Name','FA vs RAC-FA Convergence','NumberTitle','off');

semilogy(max(Curve_FA,eps),'b','LineWidth',2);
hold on;
semilogy(max(Curve_RAC,eps),'r','LineWidth',2);

grid on;
xlabel('Iteration');
ylabel('Fitness');
title('FA-ANN vs RAC-FA-ANN Training Convergence');
legend('FA-ANN','RAC-FA-ANN','Location','best');

%% =========================================================
% ANN Fitness Function
% ==========================================================

function fitness = ANN_Fitness(w,inputSize,hiddenSize,outputSize,Xtrain,Ytrain,Xval,Yval)

Ytrain_pred = ANN_Predict(w,inputSize,hiddenSize,outputSize,Xtrain);
Yval_pred = ANN_Predict(w,inputSize,hiddenSize,outputSize,Xval);

trainError = mean((Ytrain - Ytrain_pred).^2);
valError = mean((Yval - Yval_pred).^2);

fitness = 0.7*trainError + 0.3*valError;

end

%% =========================================================
% ANN Prediction Function
% ==========================================================

function Ypred = ANN_Predict(w,inputSize,hiddenSize,outputSize,X)

index = 1;

W1 = reshape(w(index:index+inputSize*hiddenSize-1),hiddenSize,inputSize);
index = index + inputSize*hiddenSize;

b1 = reshape(w(index:index+hiddenSize-1),hiddenSize,1);
index = index + hiddenSize;

W2 = reshape(w(index:index+hiddenSize*outputSize-1),outputSize,hiddenSize);
index = index + hiddenSize*outputSize;

b2 = reshape(w(index:index+outputSize-1),outputSize,1);

H = tansig(W1*X + b1);

Ypred = purelin(W2*H + b2);

end

%% =========================================================
% Performance Metrics
% ==========================================================

function [RMSE,MAE,MAPE,R2,R] = PerformanceMetrics(Yactual,Ypred)

error = Yactual - Ypred;

RMSE = sqrt(mean(error.^2));
MAE = mean(abs(error));
MAPE = mean(abs(error ./ (Yactual + eps))) * 100;

SSres = sum((Yactual - Ypred).^2);
SStot = sum((Yactual - mean(Yactual)).^2);

R2 = 1 - SSres/SStot;

Rmatrix = corrcoef(Yactual,Ypred);

if size(Rmatrix,1) > 1
    R = Rmatrix(1,2);
else
    R = NaN;
end

end

%% =========================================================
% Standard Firefly Algorithm
% ==========================================================

function [BestScore,Curve,BestPosition] = StandardFA(n,MaxIter,lb,ub,dim,fobj)

if numel(lb)==1
    lb = lb*ones(1,dim);
end

if numel(ub)==1
    ub = ub*ones(1,dim);
end

alpha = 0.25;
beta0 = 1;
gammaFA = 1;

range = ub - lb;

fireflies = lb + rand(n,dim).*range;

fitness = zeros(n,1);

for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore,idx] = min(fitness);
BestPosition = fireflies(idx,:);

Curve = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                r = norm((fireflies(i,:) - fireflies(j,:))./range);

                beta = beta0*exp(-gammaFA*r^2);

                newPosition = fireflies(i,:) ...
                    + beta*(fireflies(j,:) - fireflies(i,:)) ...
                    + alpha*(rand(1,dim)-0.5).*range;

                newPosition = BoundaryCheck(newPosition,lb,ub);

                newFitness = fobj(newPosition);

                if newFitness < fitness(i)
                    fireflies(i,:) = newPosition;
                    fitness(i) = newFitness;
                end
            end
        end
    end

    [currentBest,idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

    Curve(t) = BestScore;
end

end

%% =========================================================
% RAC-FA
% ==========================================================

function [BestScore,BestPosition,Curve] = RAC_FA(n,MaxIter,lb,ub,dim,fobj,optimum)

if numel(lb)==1
    lb = lb*ones(1,dim);
end

if numel(ub)==1
    ub = ub*ones(1,dim);
end

alpha0 = 0.35;
alpha_min = 1e-8;

beta0 = 2.0;

gamma_min = 0.001;
gamma_max = 1.0;

levy_scale0 = 0.005;
gbest_weight0 = 0.4;

stagnationLimit = 100;
restartFraction = 0.30;

range = ub - lb;

mu = 4;

chaos = zeros(n,dim);
chaos(1,:) = rand(1,dim);

for i = 2:n
    chaos(i,:) = mu .* chaos(i-1,:) .* (1 - chaos(i-1,:));
end

fireflies = lb + chaos.*range;

opp_fireflies = lb + ub - fireflies;

for i = 1:n
    if fobj(opp_fireflies(i,:)) < fobj(fireflies(i,:))
        fireflies(i,:) = opp_fireflies(i,:);
    end
end

fitness = zeros(n,1);

for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore,idx] = min(fitness);
BestPosition = fireflies(idx,:);

Curve = zeros(MaxIter,1);

stallCounter = 0;
previousBest = BestScore;

for t = 1:MaxIter

    progress = t/MaxIter;

    alpha = alpha_min + (alpha0-alpha_min)*(1-progress)^2;

    gammaFA = gamma_min + (gamma_max-gamma_min)*progress^2;

    levy_scale = levy_scale0*(1-progress);

    gbest_weight = gbest_weight0*(1-progress);

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                oldPosition = fireflies(i,:);
                oldFitness = fitness(i);

                r = norm((fireflies(i,:) - fireflies(j,:))./range);

                beta = beta0/(1 + exp(10*(r - 0.5)));

                newPosition = fireflies(i,:) ...
                    + beta*(fireflies(j,:) - fireflies(i,:)) ...
                    + gbest_weight*rand(1,dim).*(BestPosition - fireflies(i,:)) ...
                    + alpha*randn(1,dim).*range;

                if rand < 0.2*(1-progress)
                    LF = LevyFlight(dim);
                    newPosition = newPosition + levy_scale.*LF.*range;
                end

                if rand < 0.15
                    ids = randperm(n,3);
                    F = 0.5*(1-progress) + 0.1;
                    CR = 0.7;

                    mutant = fireflies(ids(1),:) + ...
                        F*(fireflies(ids(2),:) - fireflies(ids(3),:));

                    mask = rand(1,dim) < CR;

                    newPosition(mask) = mutant(mask);
                end

                newPosition = BoundaryCheck(newPosition,lb,ub);

                newFitness = fobj(newPosition);

                if newFitness < oldFitness
                    fireflies(i,:) = newPosition;
                    fitness(i) = newFitness;
                end
            end
        end
    end

    [currentBest,idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

    localStep = 0.02*range*(1-progress)^3 + 1e-10;

    for k = 1:20
        candidate = BestPosition + localStep.*randn(1,dim);
        candidate = BoundaryCheck(candidate,lb,ub);
        candidateFitness = fobj(candidate);

        if candidateFitness < BestScore
            BestScore = candidateFitness;
            BestPosition = candidate;
        end
    end

    if abs(previousBest - BestScore) < 1e-12
        stallCounter = stallCounter + 1;
    else
        stallCounter = 0;
    end

    if stallCounter > stagnationLimit

        [~,sortedIdx] = sort(fitness,'descend');

        numRestart = round(restartFraction*n);

        for rr = 1:numRestart

            id = sortedIdx(rr);

            if rand < 0.5
                fireflies(id,:) = lb + rand(1,dim).*range;
            else
                fireflies(id,:) = BestPosition + 0.2*range.*randn(1,dim);
                fireflies(id,:) = BoundaryCheck(fireflies(id,:),lb,ub);
            end

            fitness(id) = fobj(fireflies(id,:));
        end

        stallCounter = 0;
    end

    [~,worstIdx] = max(fitness);
    fireflies(worstIdx,:) = BestPosition;
    fitness(worstIdx) = BestScore;

    previousBest = BestScore;
    Curve(t) = BestScore;

    if ~isempty(optimum)
        if abs(BestScore - optimum) <= 1e-12
            Curve(t:end) = BestScore;
            break;
        end
    end
end

[BestScore,BestPosition] = FinalPolishing(BestPosition,BestScore,lb,ub,fobj);

Curve(end) = BestScore;

end

%% =========================================================
% Final Polishing
% ==========================================================

function [BestScore,BestPosition] = FinalPolishing(BestPosition,BestScore,lb,ub,fobj)

penaltyObj = @(x) PenalizedObjective(x,lb,ub,fobj);

options = optimset( ...
    'Display','off', ...
    'MaxIter',10000, ...
    'MaxFunEvals',30000, ...
    'TolX',1e-12, ...
    'TolFun',1e-12);

[xNew,~] = fminsearch(penaltyObj,BestPosition,options);

xNew = BoundaryCheck(xNew,lb,ub);

fNew = fobj(xNew);

if fNew < BestScore
    BestScore = fNew;
    BestPosition = xNew;
end

end

%% =========================================================
% Penalized Objective
% ==========================================================

function y = PenalizedObjective(x,lb,ub,fobj)

xClipped = BoundaryCheck(x,lb,ub);

penalty = sum((x - xClipped).^2)*1e6;

y = fobj(xClipped) + penalty;

end

%% =========================================================
% Levy Flight
% ==========================================================

function LF = LevyFlight(dim)

beta = 1.5;

sigma = (gamma(1+beta)*sin(pi*beta/2) / ...
    (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

u = randn(1,dim)*sigma;
v = randn(1,dim);

LF = u ./ abs(v).^(1/beta);

end

%% =========================================================
% Boundary Check
% ==========================================================

function x = BoundaryCheck(x,lb,ub)

x = max(x,lb);
x = min(x,ub);

end