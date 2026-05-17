clc; clear; close all;

%% FULL RAC-FA CODE FOR 4 BENCHMARK FUNCTIONS
% Benchmarks: Rosenbrock, Rastrigin, Ackley, Griewank

n = 50;
MaxIter = 500;
dim = 30;
runs = 10;

benchmarkFunctions = {'Rosenbrock','Rastrigin','Ackley','Griewank'};

Results = zeros(length(benchmarkFunctions),4);

for f = 1:length(benchmarkFunctions)

    funcName = benchmarkFunctions{f};
    [lb, ub, fobj, optimum] = GetBenchmarkFunction(funcName, dim);

    fprintf('\n=====================================\n');
    fprintf('Function: %s\n', funcName);
    fprintf('\n=====================================\n');

    BestScores = zeros(runs,1);
    Curves = zeros(runs,MaxIter);

    for r = 1:runs

        fprintf('Run %d/%d\n', r, runs);

        [BestScore, BestPosition, Curve] = ...
            RAC_FA(n, MaxIter, lb, ub, dim, fobj, optimum);

        BestScores(r) = BestScore;
        Curves(r,:) = Curve(:)';

    end

    Results(f,1) = min(BestScores);
    Results(f,2) = mean(BestScores);
    Results(f,3) = max(BestScores);
    Results(f,4) = std(BestScores);

    MeanCurve = mean(Curves,1);

    figure;
    semilogy(max(MeanCurve,eps),'LineWidth',2);
    grid on;
    xlabel('Iteration');
    ylabel('Best Fitness');
    title(['RAC-FA Convergence Curve - ', funcName]);

end

ResultTable = array2table(Results, ...
    'VariableNames', {'Best','Mean','Worst','Std'}, ...
    'RowNames', benchmarkFunctions);

disp(' ');
disp('FINAL RAC-FA BENCHMARK RESULTS');
disp(ResultTable);

%% =====================================================
%% BENCHMARK FUNCTIONS
%% =====================================================

function [lb, ub, fobj, optimum] = GetBenchmarkFunction(funcName, dim)

switch funcName

    case 'Rosenbrock'
        lb = -30;
        ub = 30;
        optimum = 0;
        fobj = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + ...
                        (x(1:end-1)-1).^2);

    case 'Rastrigin'
        lb = -5.12;
        ub = 5.12;
        optimum = 0;
        fobj = @(x) 10*dim + sum(x.^2 - 10*cos(2*pi*x));

    case 'Ackley'
        lb = -32;
        ub = 32;
        optimum = 0;
        fobj = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/dim)) ...
                    - exp(sum(cos(2*pi*x))/dim) + 20 + exp(1);

    case 'Griewank'
        lb = -600;
        ub = 600;
        optimum = 0;
        fobj = @(x) sum(x.^2)/4000 ...
                    - prod(cos(x./sqrt(1:dim))) + 1;

end

end

%% =====================================================
%% RAC-FA OPTIMIZER
%% =====================================================

function [BestScore, BestPosition, ConvergenceCurve] = ...
    RAC_FA(n, MaxIter, lb, ub, dim, fobj, optimum)

if nargin < 7
    optimum = [];
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

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

%% Chaotic Initialization
mu = 4;
chaos = zeros(n,dim);
chaos(1,:) = rand(1,dim);

for i = 2:n
    chaos(i,:) = mu .* chaos(i-1,:) .* (1 - chaos(i-1,:));
end

fireflies = lb + chaos .* range;

%% Opposition-Based Learning
opp_fireflies = lb + ub - fireflies;

for i = 1:n
    if fobj(opp_fireflies(i,:)) < fobj(fireflies(i,:))
        fireflies(i,:) = opp_fireflies(i,:);
    end
end

%% Fitness Evaluation
fitness = zeros(n,1);

for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore, idx] = min(fitness);
BestPosition = fireflies(idx,:);

ConvergenceCurve = zeros(MaxIter,1);
stallCounter = 0;
previousBest = BestScore;

%% Main Loop
for t = 1:MaxIter

    progress = t / MaxIter;

    alpha = alpha_min + (alpha0 - alpha_min) * (1 - progress)^2;
    gammaFA = gamma_min + (gamma_max - gamma_min) * progress^2;
    levy_scale = levy_scale0 * (1 - progress);
    gbest_weight = gbest_weight0 * (1 - progress);

    for i = 1:n

        for j = 1:n

            if fitness(j) < fitness(i)

                oldPosition = fireflies(i,:);
                oldFitness = fitness(i);

                r = norm((fireflies(i,:) - fireflies(j,:)) ./ range);

                beta = beta0 / (1 + exp(10 * (r - 0.5)));

                newPosition = fireflies(i,:) ...
                    + beta * (fireflies(j,:) - fireflies(i,:)) ...
                    + gbest_weight * rand(1,dim) .* ...
                    (BestPosition - fireflies(i,:)) ...
                    + alpha * randn(1,dim) .* range;

                %% Levy Flight
                if rand < 0.2 * (1 - progress)

                    LF = LevyFlight(dim);

                    newPosition = newPosition + levy_scale .* LF .* range;

                end

                %% Differential Mutation
                if rand < 0.15

                    ids = randperm(n,3);

                    F = 0.5 * (1 - progress) + 0.1;
                    CR = 0.7;

                    mutant = fireflies(ids(1),:) + ...
                        F * (fireflies(ids(2),:) - fireflies(ids(3),:));

                    mask = rand(1,dim) < CR;

                    newPosition(mask) = mutant(mask);

                end

                newPosition = BoundaryCheck(newPosition, lb, ub);
                newFitness = fobj(newPosition);

                if newFitness < oldFitness
                    fireflies(i,:) = newPosition;
                    fitness(i) = newFitness;
                else
                    fireflies(i,:) = oldPosition;
                    fitness(i) = oldFitness;
                end

            end

        end

    end

    %% Update Best
    [currentBest, idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

    %% Local Refinement
    localStep = 0.02 * range * (1 - progress)^3 + 1e-10;

    for k = 1:20

        candidate = BestPosition + localStep .* randn(1,dim);
        candidate = BoundaryCheck(candidate, lb, ub);

        candidateFitness = fobj(candidate);

        if candidateFitness < BestScore
            BestScore = candidateFitness;
            BestPosition = candidate;
        end

    end

    %% Stagnation Check
    if abs(previousBest - BestScore) < 1e-12
        stallCounter = stallCounter + 1;
    else
        stallCounter = 0;
    end

    %% Restart Mechanism
    if stallCounter > stagnationLimit

        [~, sortedIdx] = sort(fitness,'descend');
        numRestart = round(restartFraction * n);

        for rr = 1:numRestart

            id = sortedIdx(rr);

            if rand < 0.5
                fireflies(id,:) = lb + rand(1,dim) .* range;
            else
                fireflies(id,:) = BestPosition + ...
                    0.2 * range .* randn(1,dim);

                fireflies(id,:) = BoundaryCheck(fireflies(id,:), lb, ub);
            end

            fitness(id) = fobj(fireflies(id,:));

        end

        stallCounter = 0;

    end

    %% Elitism
    [~, worstIdx] = max(fitness);

    fireflies(worstIdx,:) = BestPosition;
    fitness(worstIdx) = BestScore;

    previousBest = BestScore;
    ConvergenceCurve(t) = BestScore;

    %% Early Stopping
    if ~isempty(optimum)
        if abs(BestScore - optimum) <= 1e-12
            ConvergenceCurve(t:end) = BestScore;
            break;
        end
    end

end

%% Final Polishing
[BestScore, BestPosition] = ...
    FinalPolishing(BestPosition, BestScore, lb, ub, fobj);

ConvergenceCurve(end) = BestScore;

end

%% =====================================================
%% FINAL LOCAL POLISHING
%% =====================================================

function [BestScore, BestPosition] = ...
    FinalPolishing(BestPosition, BestScore, lb, ub, fobj)

penaltyObj = @(x) PenalizedObjective(x, lb, ub, fobj);

options = optimset( ...
    'Display','off', ...
    'MaxIter',20000, ...
    'MaxFunEvals',50000, ...
    'TolX',1e-14, ...
    'TolFun',1e-14);

[xNew, ~] = fminsearch(penaltyObj, BestPosition, options);

xNew = BoundaryCheck(xNew, lb, ub);
fNew = fobj(xNew);

if fNew < BestScore
    BestScore = fNew;
    BestPosition = xNew;
end

end

%% =====================================================
%% PENALIZED OBJECTIVE
%% =====================================================

function y = PenalizedObjective(x, lb, ub, fobj)

xClipped = BoundaryCheck(x, lb, ub);
penalty = sum((x - xClipped).^2) * 1e6;
y = fobj(xClipped) + penalty;

end

%% =====================================================
%% LEVY FLIGHT
%% =====================================================

function LF = LevyFlight(dim)

beta = 1.5;

sigma = (gamma(1+beta) * sin(pi*beta/2) / ...
    (gamma((1+beta)/2) * beta * ...
    2^((beta-1)/2)))^(1/beta);

u = randn(1,dim) * sigma;
v = randn(1,dim);

LF = u ./ abs(v).^(1/beta);

end

%% =====================================================
%% BOUNDARY CHECK
%% =====================================================

function x = BoundaryCheck(x, lb, ub)

x = max(x, lb);
x = min(x, ub);

end