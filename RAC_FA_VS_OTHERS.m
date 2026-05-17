clc;
clear;
close all;

%% =====================================================
% COMPARISON OF FA, RAC-FA, PSO, GWO AND SMO
% Optimization Problem: Rosenbrock Function
% =====================================================

SearchAgents = 50;
MaxIter = 1000;
dim = 30;

lb = -5;
ub = 10;

fobj = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + ...
                (x(1:end-1)-1).^2);

optimum = 0;

%% Run Algorithms

[best_FA, curve_FA] = StandardFA(SearchAgents, MaxIter, lb, ub, dim, fobj);

[best_RACFA, ~, curve_RACFA] = RAC_FA(SearchAgents, MaxIter, lb, ub, dim, fobj, optimum);

[best_PSO, curve_PSO] = PSO(SearchAgents, MaxIter, lb, ub, dim, fobj);

[best_GWO, curve_GWO] = GWO(SearchAgents, MaxIter, lb, ub, dim, fobj);

[best_SMO, curve_SMO] = SMO(SearchAgents, MaxIter, lb, ub, dim, fobj);

%% Display Results

fprintf('\n=========================================\n');
fprintf('Optimization Problem: Rosenbrock Function\n');
fprintf('Dimension: %d\n', dim);
fprintf('Search Range: [%g, %g]\n', lb, ub);
fprintf('Global Optimum: %g\n', optimum);
fprintf('=========================================\n');

fprintf('FA Best Fitness      = %.10f\n', best_FA);
fprintf('RAC-FA Best Fitness  = %.10f\n', best_RACFA);
fprintf('PSO Best Fitness     = %.10f\n', best_PSO);
fprintf('GWO Best Fitness     = %.10f\n', best_GWO);
fprintf('SMO Best Fitness     = %.10f\n', best_SMO);

%% Results Table

Algorithm = {'FA'; 'RAC-FA'; 'PSO'; 'GWO'; 'SMO'};
BestFitness = [best_FA; best_RACFA; best_PSO; best_GWO; best_SMO];
Error = abs(BestFitness - optimum);

ResultTable = table(Algorithm, BestFitness, Error);

disp(' ');
disp('================ SUMMARY TABLE ================');
disp(ResultTable);

%% Plot Convergence Curves

figure('Name','FA vs RAC-FA vs PSO vs GWO vs SMO','NumberTitle','off');

semilogy(max(curve_FA,eps),'r','LineWidth',2);
hold on;
semilogy(max(curve_RACFA,eps),'b','LineWidth',2);
semilogy(max(curve_PSO,eps),'g','LineWidth',2);
semilogy(max(curve_GWO,eps),'m','LineWidth',2);
semilogy(max(curve_SMO,eps),'k','LineWidth',2);

grid on;
xlabel('Iteration');
ylabel('Best Fitness');
title('Performance Comparison on Rosenbrock Function');

legend('FA','RAC-FA','PSO','GWO','SMO');

%% =====================================================
% STANDARD FIREFLY ALGORITHM
% =====================================================

function [BestScore, Curve] = StandardFA(n, MaxIter, lb, ub, dim, fobj)

alpha = 0.25;
beta0 = 1;
gammaFA = 1;

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

fireflies = lb + rand(n,dim).*range;

fitness = zeros(n,1);

for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore, idx] = min(fitness);
BestPosition = fireflies(idx,:);

Curve = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                r = norm((fireflies(i,:) - fireflies(j,:))./range);

                beta = beta0 * exp(-gammaFA*r^2);

                newPosition = fireflies(i,:) ...
                    + beta*(fireflies(j,:) - fireflies(i,:)) ...
                    + alpha*(rand(1,dim)-0.5).*range;

                newPosition = BoundaryCheck(newPosition, lb, ub);

                newFitness = fobj(newPosition);

                if newFitness < fitness(i)
                    fireflies(i,:) = newPosition;
                    fitness(i) = newFitness;
                end
            end
        end
    end

    [currentBest, idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

    Curve(t) = BestScore;
end

end

%% =====================================================
% RAC-FA
% =====================================================

function [BestScore, BestPosition, Curve] = RAC_FA(n, MaxIter, lb, ub, dim, fobj, optimum)

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

[BestScore, idx] = min(fitness);
BestPosition = fireflies(idx,:);

Curve = zeros(MaxIter,1);

stallCounter = 0;
previousBest = BestScore;

for t = 1:MaxIter

    progress = t / MaxIter;

    alpha = alpha_min + (alpha0 - alpha_min)*(1-progress)^2;
    gammaFA = gamma_min + (gamma_max-gamma_min)*progress^2;

    levy_scale = levy_scale0*(1-progress);
    gbest_weight = gbest_weight0*(1-progress);

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                oldPosition = fireflies(i,:);
                oldFitness = fitness(i);

                r = norm((fireflies(i,:) - fireflies(j,:))./range);

                beta = beta0 / (1 + exp(10*(r - 0.5)));

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

                newPosition = BoundaryCheck(newPosition, lb, ub);

                newFitness = fobj(newPosition);

                if newFitness < oldFitness
                    fireflies(i,:) = newPosition;
                    fitness(i) = newFitness;
                end
            end
        end
    end

    [currentBest, idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

    localStep = 0.02*range*(1-progress)^3 + 1e-10;

    for k = 1:20

        candidate = BestPosition + localStep.*randn(1,dim);
        candidate = BoundaryCheck(candidate, lb, ub);
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

        [~, sortedIdx] = sort(fitness,'descend');

        numRestart = round(restartFraction*n);

        for rr = 1:numRestart

            id = sortedIdx(rr);

            if rand < 0.5
                fireflies(id,:) = lb + rand(1,dim).*range;
            else
                fireflies(id,:) = BestPosition + 0.2*range.*randn(1,dim);
                fireflies(id,:) = BoundaryCheck(fireflies(id,:), lb, ub);
            end

            fitness(id) = fobj(fireflies(id,:));
        end

        stallCounter = 0;
    end

    [~, worstIdx] = max(fitness);
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

[BestScore, BestPosition] = FinalPolishing(BestPosition, BestScore, lb, ub, fobj);
Curve(end) = BestScore;

end

%% =====================================================
% PARTICLE SWARM OPTIMIZATION
% =====================================================

function [BestScore, Curve] = PSO(n, MaxIter, lb, ub, dim, fobj)

w = 0.7;
c1 = 1.5;
c2 = 1.5;

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

X = lb + rand(n,dim).*range;
V = zeros(n,dim);

Pbest = X;
PbestFit = zeros(n,1);

for i = 1:n
    PbestFit(i) = fobj(X(i,:));
end

[BestScore, idx] = min(PbestFit);
Gbest = Pbest(idx,:);

Curve = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n

        V(i,:) = w*V(i,:) ...
            + c1*rand(1,dim).*(Pbest(i,:) - X(i,:)) ...
            + c2*rand(1,dim).*(Gbest - X(i,:));

        X(i,:) = X(i,:) + V(i,:);

        X(i,:) = BoundaryCheck(X(i,:), lb, ub);

        fit = fobj(X(i,:));

        if fit < PbestFit(i)
            Pbest(i,:) = X(i,:);
            PbestFit(i) = fit;
        end
    end

    [currentBest, idx] = min(PbestFit);

    if currentBest < BestScore
        BestScore = currentBest;
        Gbest = Pbest(idx,:);
    end

    Curve(t) = BestScore;
end

end

%% =====================================================
% GREY WOLF OPTIMIZER
% =====================================================

function [BestScore, Curve] = GWO(n, MaxIter, lb, ub, dim, fobj)

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

X = lb + rand(n,dim).*range;

Alpha_score = inf;
Beta_score = inf;
Delta_score = inf;

Alpha_pos = zeros(1,dim);
Beta_pos = zeros(1,dim);
Delta_pos = zeros(1,dim);

Curve = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n

        X(i,:) = BoundaryCheck(X(i,:), lb, ub);

        fitness = fobj(X(i,:));

        if fitness < Alpha_score
            Alpha_score = fitness;
            Alpha_pos = X(i,:);
        elseif fitness < Beta_score
            Beta_score = fitness;
            Beta_pos = X(i,:);
        elseif fitness < Delta_score
            Delta_score = fitness;
            Delta_pos = X(i,:);
        end
    end

    a = 2 - t*(2/MaxIter);

    for i = 1:n
        for d = 1:dim

            r1 = rand; r2 = rand;
            A1 = 2*a*r1 - a;
            C1 = 2*r2;

            D_alpha = abs(C1*Alpha_pos(d) - X(i,d));
            X1 = Alpha_pos(d) - A1*D_alpha;

            r1 = rand; r2 = rand;
            A2 = 2*a*r1 - a;
            C2 = 2*r2;

            D_beta = abs(C2*Beta_pos(d) - X(i,d));
            X2 = Beta_pos(d) - A2*D_beta;

            r1 = rand; r2 = rand;
            A3 = 2*a*r1 - a;
            C3 = 2*r2;

            D_delta = abs(C3*Delta_pos(d) - X(i,d));
            X3 = Delta_pos(d) - A3*D_delta;

            X(i,d) = (X1 + X2 + X3)/3;
        end
    end

    Curve(t) = Alpha_score;
end

BestScore = Alpha_score;

end

%% =====================================================
% SIMPLIFIED SPIDER MONKEY OPTIMIZATION
% =====================================================

function [BestScore, Curve] = SMO(n, MaxIter, lb, ub, dim, fobj)

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

X = lb + rand(n,dim).*range;

fitness = zeros(n,1);

for i = 1:n
    fitness(i) = fobj(X(i,:));
end

[BestScore, idx] = min(fitness);
GlobalLeader = X(idx,:);

Curve = zeros(MaxIter,1);

pr = 0.3;

for t = 1:MaxIter

    for i = 1:n

        randIndex = randi(n);

        newPosition = X(i,:);

        for d = 1:dim

            if rand < pr

                phi = -1 + 2*rand;

                newPosition(d) = X(i,d) ...
                    + phi*(GlobalLeader(d) - X(i,d)) ...
                    + phi*(X(randIndex,d) - X(i,d));
            end
        end

        newPosition = BoundaryCheck(newPosition, lb, ub);

        newFitness = fobj(newPosition);

        if newFitness < fitness(i)
            X(i,:) = newPosition;
            fitness(i) = newFitness;
        end
    end

    [currentBest, idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        GlobalLeader = X(idx,:);
    end

    Curve(t) = BestScore;
end

end

%% =====================================================
% SUPPORT FUNCTIONS
% =====================================================

function [BestScore, BestPosition] = FinalPolishing(BestPosition, BestScore, lb, ub, fobj)

penaltyObj = @(x) PenalizedObjective(x, lb, ub, fobj);

options = optimset( ...
    'Display','off', ...
    'MaxIter',10000, ...
    'MaxFunEvals',30000, ...
    'TolX',1e-12, ...
    'TolFun',1e-12);

[xNew, ~] = fminsearch(penaltyObj, BestPosition, options);

xNew = BoundaryCheck(xNew, lb, ub);

fNew = fobj(xNew);

if fNew < BestScore
    BestScore = fNew;
    BestPosition = xNew;
end

end

function y = PenalizedObjective(x, lb, ub, fobj)

xClipped = BoundaryCheck(x, lb, ub);
penalty = sum((x - xClipped).^2)*1e6;

y = fobj(xClipped) + penalty;

end

function LF = LevyFlight(dim)

beta = 1.5;

sigma = (gamma(1+beta)*sin(pi*beta/2) / ...
    (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

u = randn(1,dim)*sigma;
v = randn(1,dim);

LF = u ./ abs(v).^(1/beta);

end

function x = BoundaryCheck(x, lb, ub)

x = max(x, lb);
x = min(x, ub);

end