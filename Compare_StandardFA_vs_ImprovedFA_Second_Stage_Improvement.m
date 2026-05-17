clc;
clear;
close all;

%% =====================================================
% Standard FA vs Proposed Improved FA
% Improved FA includes:
% 1. Chaotic initialization
% 2. Sigmoid attraction
% 3. Adaptive alpha
% 4. Dynamic gamma
% 5. Levy flight
% 6. Elitism
% =====================================================

SearchAgents = 30;
MaxIter = 200;
dim = 30;

benchmarks = {'Sphere','Rastrigin','Ackley','Rosenbrock'};

figure('Name','Standard FA vs Proposed Improved FA','NumberTitle','off');

for b = 1:length(benchmarks)

    [lb, ub, fobj] = GetBenchmark(benchmarks{b}, dim);

    [best_std, curve_std] = StandardFA(SearchAgents, MaxIter, lb, ub, dim, fobj);
    [best_imp, curve_imp] = ImprovedFA(SearchAgents, MaxIter, lb, ub, dim, fobj);

    fprintf('\n==============================\n');
    fprintf('Benchmark Function: %s\n', benchmarks{b});
    fprintf('Ideal Best Fitness: 0\n');
    fprintf('Standard FA Best Fitness: %.10f\n', best_std);
    fprintf('Improved FA Best Fitness: %.10f\n', best_imp);

    subplot(2,2,b)
    semilogy(curve_std,'r','LineWidth',2);
    hold on;
    semilogy(curve_imp,'b','LineWidth',2);
    yline(1e-12,'k--','Ideal','LineWidth',1);
    grid on;
    xlabel('Iteration');
    ylabel('Best Fitness');
    title(benchmarks{b});
    legend('Standard FA','Improved FA','Ideal');
end

sgtitle('Comparison of Standard FA and Proposed Improved FA');


%% =====================================================
% Standard Firefly Algorithm
% =====================================================

function [BestScore, ConvergenceCurve] = StandardFA(n, MaxIter, lb, ub, dim, fobj)

alpha = 0.25;
beta0 = 1;
gamma = 1;

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);

fireflies = rand(n,dim).*(ub-lb) + lb;

fitness = zeros(n,1);
for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore, bestIndex] = min(fitness);
BestPosition = fireflies(bestIndex,:);

ConvergenceCurve = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                r = norm(fireflies(i,:) - fireflies(j,:));
                beta = beta0 * exp(-gamma * r^2);

                fireflies(i,:) = fireflies(i,:) ...
                    + beta * (fireflies(j,:) - fireflies(i,:)) ...
                    + alpha * (rand(1,dim) - 0.5);

                fireflies(i,:) = BoundaryCheck(fireflies(i,:), lb, ub);
                fitness(i) = fobj(fireflies(i,:));
            end
        end
    end

    [currentBest, bestIndex] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(bestIndex,:);
    end

    ConvergenceCurve(t) = BestScore;
end

end


%% =====================================================
% Proposed Improved Firefly Algorithm
% =====================================================

function [BestScore, ConvergenceCurve] = ImprovedFA(n, MaxIter, lb, ub, dim, fobj)

alpha0 = 0.5;
alpha_min = 0.001;

beta0 = 2;

gamma_min = 0.05;
gamma_max = 1.5;

r0 = 1;

levy_prob = 0.25;
levy_scale = 0.01;

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);

%% Chaotic Initialization using Logistic Map
mu = 4;
chaos = zeros(n,dim);
chaos(1,:) = rand(1,dim);

for i = 2:n
    chaos(i,:) = mu .* chaos(i-1,:) .* (1 - chaos(i-1,:));
end

fireflies = lb + chaos .* (ub - lb);

%% Opposition-Based Learning Initialization
opp_fireflies = lb + ub - fireflies;

for i = 1:n
    fit1 = fobj(fireflies(i,:));
    fit2 = fobj(opp_fireflies(i,:));

    if fit2 < fit1
        fireflies(i,:) = opp_fireflies(i,:);
    end
end

fitness = zeros(n,1);
for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore, bestIndex] = min(fitness);
BestPosition = fireflies(bestIndex,:);

ConvergenceCurve = zeros(MaxIter,1);

%% Main Loop
for t = 1:MaxIter

    % Adaptive alpha: high at beginning, low at end
    alpha = alpha_min + (alpha0 - alpha_min) * exp(-4*t/MaxIter);

    % Dynamic gamma: increases with iteration
    gamma = gamma_min + (gamma_max - gamma_min) * (t/MaxIter);

    % Chaotic movement variable
    c = rand;
    c = mu * c * (1 - c);

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                r = norm(fireflies(i,:) - fireflies(j,:));

                % Sigmoid attraction
                beta = beta0 / (1 + exp(gamma * (r - r0)));

                fireflies(i,:) = fireflies(i,:) ...
                    + beta * (fireflies(j,:) - fireflies(i,:)) ...
                    + alpha * (c - 0.5) .* (ub - lb);

                % Levy flight for escaping local optima
                if rand < levy_prob
                    LF = LevyFlight(dim);
                    fireflies(i,:) = fireflies(i,:) ...
                        + levy_scale .* LF .* (fireflies(i,:) - BestPosition);
                end

                fireflies(i,:) = BoundaryCheck(fireflies(i,:), lb, ub);
                fitness(i) = fobj(fireflies(i,:));
            end
        end
    end

    %% Elitism
    [currentBest, bestIndex] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(bestIndex,:);
    else
        [~, worstIndex] = max(fitness);
        fireflies(worstIndex,:) = BestPosition;
        fitness(worstIndex) = BestScore;
    end

    %% Local refinement around best solution
    refine_step = 0.001 * (ub - lb) * (1 - t/MaxIter);
    candidate = BestPosition + refine_step .* randn(1,dim);
    candidate = BoundaryCheck(candidate, lb, ub);

    candidateFitness = fobj(candidate);

    if candidateFitness < BestScore
        BestScore = candidateFitness;
        BestPosition = candidate;
    end

    ConvergenceCurve(t) = BestScore;
end

end


%% =====================================================
% Levy Flight Function
% =====================================================

function LF = LevyFlight(dim)

beta = 1.5;

sigma = (gamma(1+beta) * sin(pi*beta/2) / ...
        (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);

u = randn(1,dim) * sigma;
v = randn(1,dim);

LF = u ./ abs(v).^(1/beta);

end


%% =====================================================
% Boundary Check
% =====================================================

function x = BoundaryCheck(x, lb, ub)

x = max(x, lb);
x = min(x, ub);

end


%% =====================================================
% Benchmark Functions
% =====================================================

function [lb, ub, fobj] = GetBenchmark(name, dim)

switch name

    case 'Sphere'
        lb = -100;
        ub = 100;
        fobj = @(x) sum(x.^2);

    case 'Rastrigin'
        lb = -5.12;
        ub = 5.12;
        fobj = @(x) 10*dim + sum(x.^2 - 10*cos(2*pi*x));

    case 'Ackley'
        lb = -32;
        ub = 32;
        fobj = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/dim)) ...
                    - exp(sum(cos(2*pi*x))/dim) + 20 + exp(1);

    case 'Rosenbrock'
        lb = -30;
        ub = 30;
        fobj = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 ...
                    + (x(1:end-1)-1).^2);

    otherwise
        error('Unknown benchmark function');
end

end