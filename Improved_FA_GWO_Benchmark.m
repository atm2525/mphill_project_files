clc; clear; close all;

%% UPDATED HYBRID FA-GWO BENCHMARK
SearchAgents = 50;
MaxIter = 300;
dim = 30;
runs = 20;

benchmarkFunctions = {'Sphere','Rastrigin','Ackley','Rosenbrock','Griewank'};

Results = [];

for f = 1:length(benchmarkFunctions)

    funcName = benchmarkFunctions{f};
    [lb, ub, fobj] = GetFunctionDetails(funcName, dim);

    fprintf('\n==============================\n');
    fprintf('Function: %s\n', funcName);
    fprintf('==============================\n');

    PSO_scores = zeros(runs,1);
    GWO_scores = zeros(runs,1);
    FA_scores = zeros(runs,1);
    IFAGWO_scores = zeros(runs,1);

    for r = 1:runs
        fprintf('Run %d/%d\n', r, runs);

        [PSO_scores(r), PSO_curve] = PSO(SearchAgents, MaxIter, lb, ub, dim, fobj);
        [GWO_scores(r), GWO_curve] = GWO(SearchAgents, MaxIter, lb, ub, dim, fobj);
        [FA_scores(r), FA_curve] = FA(SearchAgents, MaxIter, lb, ub, dim, fobj);
        [IFAGWO_scores(r), IFAGWO_curve] = Improved_FA_GWO(SearchAgents, MaxIter, lb, ub, dim, fobj);
    end

    Results = [Results;
        mean(PSO_scores), mean(GWO_scores), mean(FA_scores), mean(IFAGWO_scores)];

    fprintf('\nAverage Results for %s\n', funcName);
    fprintf('PSO        = %.6e\n', mean(PSO_scores));
    fprintf('GWO        = %.6e\n', mean(GWO_scores));
    fprintf('FA         = %.6e\n', mean(FA_scores));
    fprintf('FA-GWO     = %.6e\n', mean(IFAGWO_scores));

    figure;
    semilogy(PSO_curve, 'LineWidth', 2); hold on;
    semilogy(GWO_curve, 'LineWidth', 2);
    semilogy(FA_curve, 'LineWidth', 2);
    semilogy(IFAGWO_curve, 'LineWidth', 2);
    grid on;
    xlabel('Iteration');
    ylabel('Best Fitness');
    title(['Convergence Curve - ', funcName]);
    legend('PSO','GWO','FA','Improved FA-GWO');
end

ResultTable = array2table(Results, ...
    'VariableNames', {'PSO','GWO','FA','Improved_FA_GWO'}, ...
    'RowNames', benchmarkFunctions);

disp(' ');
disp('FINAL AVERAGE FITNESS RESULTS');
disp(ResultTable);

%% ============================================================
%% BENCHMARK FUNCTIONS
%% ============================================================

function [lb, ub, fobj] = GetFunctionDetails(F, dim)

switch F
    case 'Sphere'
        lb = -100; ub = 100;
        fobj = @(x) sum(x.^2);

    case 'Rastrigin'
        lb = -5.12; ub = 5.12;
        fobj = @(x) 10*dim + sum(x.^2 - 10*cos(2*pi*x));

    case 'Ackley'
        lb = -32; ub = 32;
        fobj = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/dim)) ...
                    - exp(sum(cos(2*pi*x))/dim) + 20 + exp(1);

    case 'Rosenbrock'
        lb = -30; ub = 30;
        fobj = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 + ...
                    (x(1:end-1)-1).^2);

    case 'Griewank'
        lb = -600; ub = 600;
        fobj = @(x) sum(x.^2)/4000 - prod(cos(x./sqrt(1:dim))) + 1;
end

end

%% ============================================================
%% PSO
%% ============================================================

function [BestScore, Convergence] = PSO(n, MaxIter, lb, ub, dim, fobj)

w = 0.7;
c1 = 1.5;
c2 = 1.5;

X = lb + rand(n,dim).*(ub-lb);
V = zeros(n,dim);

pBest = X;
pBestScore = inf(n,1);

gBest = zeros(1,dim);
gBestScore = inf;

Convergence = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n
        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);

        fitness = fobj(X(i,:));

        if fitness < pBestScore(i)
            pBestScore(i) = fitness;
            pBest(i,:) = X(i,:);
        end

        if fitness < gBestScore
            gBestScore = fitness;
            gBest = X(i,:);
        end
    end

    for i = 1:n
        V(i,:) = w*V(i,:) ...
               + c1*rand(1,dim).*(pBest(i,:) - X(i,:)) ...
               + c2*rand(1,dim).*(gBest - X(i,:));

        X(i,:) = X(i,:) + V(i,:);
    end

    Convergence(t) = gBestScore;
end

BestScore = gBestScore;

end

%% ============================================================
%% GWO
%% ============================================================

function [Alpha_score, Convergence] = GWO(n, MaxIter, lb, ub, dim, fobj)

X = lb + rand(n,dim).*(ub-lb);

Alpha_pos = zeros(1,dim);
Beta_pos = zeros(1,dim);
Delta_pos = zeros(1,dim);

Alpha_score = inf;
Beta_score = inf;
Delta_score = inf;

Convergence = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n
        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);

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
        for j = 1:dim

            r1 = rand; r2 = rand;
            A1 = 2*a*r1 - a;
            C1 = 2*r2;
            D_alpha = abs(C1*Alpha_pos(j) - X(i,j));
            X1 = Alpha_pos(j) - A1*D_alpha;

            r1 = rand; r2 = rand;
            A2 = 2*a*r1 - a;
            C2 = 2*r2;
            D_beta = abs(C2*Beta_pos(j) - X(i,j));
            X2 = Beta_pos(j) - A2*D_beta;

            r1 = rand; r2 = rand;
            A3 = 2*a*r1 - a;
            C3 = 2*r2;
            D_delta = abs(C3*Delta_pos(j) - X(i,j));
            X3 = Delta_pos(j) - A3*D_delta;

            X(i,j) = (X1 + X2 + X3)/3;
        end
    end

    Convergence(t) = Alpha_score;
end

end

%% ============================================================
%% STANDARD FIREFLY ALGORITHM
%% ============================================================

function [BestScore, Convergence] = FA(n, MaxIter, lb, ub, dim, fobj)

alpha = 0.2;
beta0 = 2;
gamma = 1/dim;

X = lb + rand(n,dim).*(ub-lb);
fitness = zeros(n,1);
Convergence = zeros(MaxIter,1);

for i = 1:n
    fitness(i) = fobj(X(i,:));
end

[BestScore, idx] = min(fitness);
BestPos = X(idx,:);

for t = 1:MaxIter

    alpha_t = alpha * (1 - t/MaxIter);

    for i = 1:n
        Xi_old = X(i,:);
        oldFitness = fitness(i);

        for j = 1:n
            if fitness(j) < fitness(i)

                rij = norm(X(i,:) - X(j,:));
                beta = beta0 * exp(-gamma * rij^2);

                X(i,:) = X(i,:) ...
                       + beta * rand(1,dim).*(X(j,:) - X(i,:)) ...
                       + alpha_t * (rand(1,dim)-0.5).*(ub-lb);
            end
        end

        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);

        newFitness = fobj(X(i,:));

        if newFitness < oldFitness
            fitness(i) = newFitness;
        else
            X(i,:) = Xi_old;
        end

        if fitness(i) < BestScore
            BestScore = fitness(i);
            BestPos = X(i,:);
        end
    end

    Convergence(t) = BestScore;
end

end

%% ============================================================
%% IMPROVED FIREFLY ALGORITHM USING GWO
%% ============================================================

function [BestScore, Convergence] = Improved_FA_GWO(n, MaxIter, lb, ub, dim, fobj)

alpha = 0.2;
beta0 = 2;
gamma = 1/dim;

X = lb + rand(n,dim).*(ub-lb);
fitness = zeros(n,1);
Convergence = zeros(MaxIter,1);

for i = 1:n
    fitness(i) = fobj(X(i,:));
end

for t = 1:MaxIter

    [fitness, index] = sort(fitness);
    X = X(index,:);

    Alpha = X(1,:);
    Beta  = X(2,:);
    Delta = X(3,:);

    BestScore = fitness(1);

    a = 2 - 2*(t/MaxIter);
    alpha_t = alpha * (1 - t/MaxIter);
    lambda = 1.5 * (1 - t/MaxIter);

    for i = 1:n

        Xi_old = X(i,:);
        oldFitness = fitness(i);

        %% FA movement
        for j = 1:n
            if fitness(j) < fitness(i)

                rij = norm(X(i,:) - X(j,:));
                beta_ij = beta0 * exp(-gamma * rij^2);

                X(i,:) = X(i,:) ...
                       + beta_ij * rand(1,dim).*(X(j,:) - X(i,:)) ...
                       + alpha_t * (rand(1,dim)-0.5).*(ub-lb);
            end
        end

        %% GWO guidance
        A1 = 2*a*rand(1,dim) - a;
        C1 = 2*rand(1,dim);
        D_alpha = abs(C1.*Alpha - X(i,:));
        X1 = Alpha - A1.*D_alpha;

        A2 = 2*a*rand(1,dim) - a;
        C2 = 2*rand(1,dim);
        D_beta = abs(C2.*Beta - X(i,:));
        X2 = Beta - A2.*D_beta;

        A3 = 2*a*rand(1,dim) - a;
        C3 = 2*rand(1,dim);
        D_delta = abs(C3.*Delta - X(i,:));
        X3 = Delta - A3.*D_delta;

        GWO_position = (X1 + X2 + X3)/3;

        %% Hybrid FA-GWO movement
        X(i,:) = (1-lambda)*X(i,:) + lambda*GWO_position;

        X(i,:) = max(X(i,:), lb);
        X(i,:) = min(X(i,:), ub);

        newFitness = fobj(X(i,:));

        %% Greedy selection
        if newFitness < oldFitness
            fitness(i) = newFitness;
        else
            X(i,:) = Xi_old;
            fitness(i) = oldFitness;
        end
    end

    [BestScore, bestIndex] = min(fitness);
    Convergence(t) = BestScore;

end

end