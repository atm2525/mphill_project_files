clc;
clear;
close all;

%% =====================================================
% Standard FA vs Robust Improved FA
% Benchmark Functions from Table 4.1
% Corrected Levy N.13 and Schwefel
% =====================================================

SearchAgents = 100;
MaxIter = 3000;

functions = {
    'Michalewicz',              0,       pi,       -9.66015, 10, 1e-5;
    'Ackley',                  -1,        1,        0,       30, 1e-5;
    'Levy1',                  -10,       10,        0,       30, 1e-5;
    'LevyN13',                 -5,        5,        0,        2, 1e-5;
    'Colville',               -10,       10,        0,        4, 1e-5;
    'Schwefel',              -500,      500,        0,        2, 1e-5;
    'Beale',                 -4.5,      4.5,        0,        2, 1e-5;
    'Rosenbrock',              -5,       10,        0,       30, 1e-1;
    'RotatedHyperEllipsoid', -65.536,  65.536,      0,        2, 1e-5;
};

figure('Name','Standard FA vs Robust Improved FA','NumberTitle','off');

Results = [];

for b = 1:size(functions,1)

    fname = functions{b,1};
    lb = functions{b,2};
    ub = functions{b,3};
    optimum = functions{b,4};
    dim = functions{b,5};
    errorLimit = functions{b,6};

    fobj = GetTestFunction(fname, dim);

    [best_std, curve_std] = StandardFA(SearchAgents, MaxIter, lb, ub, dim, fobj);
    [best_imp, curve_imp] = ImprovedFA(SearchAgents, MaxIter, lb, ub, dim, fobj, optimum);

    err_std = abs(best_std - optimum);
    err_imp = abs(best_imp - optimum);

    if err_imp <= errorLimit
        status = "PASS";
    else
        status = "FAIL";
    end

    fprintf('\n=========================================\n');
    fprintf('Function: %s\n', fname);
    fprintf('Dimension: %d\n', dim);
    fprintf('Search Range: [%g, %g]\n', lb, ub);
    fprintf('Optimum Value: %.10f\n', optimum);
    fprintf('Error Limit: %.1E\n', errorLimit);
    fprintf('Standard FA Best Fitness: %.10f\n', best_std);
    fprintf('Standard FA Error: %.10e\n', err_std);
    fprintf('Improved FA Best Fitness: %.10f\n', best_imp);
    fprintf('Improved FA Error: %.10e\n', err_imp);
    fprintf('Improved FA Status: %s\n', status);

    Results = [Results; b, best_std, err_std, best_imp, err_imp];

    subplot(3,3,b)
    semilogy(max(abs(curve_std - optimum),eps),'r','LineWidth',2);
    hold on;
    semilogy(max(abs(curve_imp - optimum),eps),'b','LineWidth',2);
    yline(errorLimit,'k--','Error Limit','LineWidth',1);
    grid on;
    xlabel('Iteration');
    ylabel('Absolute Error');
    title(fname,'Interpreter','none');
    legend('Standard FA','Improved FA','Error Limit');
end

sgtitle('Standard FA vs Robust Improved FA with Final Polishing');

ResultTable = table(Results(:,1), functions(:,1), Results(:,2), Results(:,3), ...
    Results(:,4), Results(:,5), ...
    'VariableNames', {'SN','Function_Name','Standard_Best','Standard_Error', ...
    'Improved_Best','Improved_Error'});

disp(' ');
disp('================ SUMMARY TABLE ================');
disp(ResultTable);


%% =====================================================
% Standard Firefly Algorithm
% =====================================================

function [BestScore, ConvergenceCurve] = StandardFA(n, MaxIter, lb, ub, dim, fobj)

alpha = 0.25;
beta0 = 1;
gammaFA = 1;

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

fireflies = rand(n,dim).*range + lb;

fitness = zeros(n,1);
for i = 1:n
    fitness(i) = fobj(fireflies(i,:));
end

[BestScore, idx] = min(fitness);
BestPosition = fireflies(idx,:);

ConvergenceCurve = zeros(MaxIter,1);

for t = 1:MaxIter

    for i = 1:n
        for j = 1:n

            if fitness(j) < fitness(i)

                r = norm((fireflies(i,:) - fireflies(j,:))./range);
                beta = beta0 * exp(-gammaFA * r^2);

                newPosition = fireflies(i,:) ...
                    + beta * (fireflies(j,:) - fireflies(i,:)) ...
                    + alpha * (rand(1,dim)-0.5).*range;

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

    ConvergenceCurve(t) = BestScore;
end

end


%% =====================================================
% Robust Improved Firefly Algorithm
% =====================================================

function [BestScore, ConvergenceCurve] = ImprovedFA(n, MaxIter, lb, ub, dim, fobj, optimum)

alpha0 = 0.35;
alpha_min = 1e-8;

beta0 = 2.0;
gamma_min = 0.001;
gamma_max = 1.0;

levy_scale0 = 0.005;
gbest_weight0 = 0.4;

lb = lb .* ones(1,dim);
ub = ub .* ones(1,dim);
range = ub - lb;

mu = 4;
chaos = zeros(n,dim);
chaos(1,:) = rand(1,dim);

for i = 2:n
    chaos(i,:) = mu .* chaos(i-1,:) .* (1 - chaos(i-1,:));
end

fireflies = lb + chaos .* range;

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

ConvergenceCurve = zeros(MaxIter,1);

stallCounter = 0;
previousBest = BestScore;

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

                r = norm((fireflies(i,:) - fireflies(j,:))./range);

                beta = beta0 / (1 + exp(10*(r - 0.5)));

                newPosition = fireflies(i,:) ...
                    + beta * (fireflies(j,:) - fireflies(i,:)) ...
                    + gbest_weight * rand(1,dim) .* (BestPosition - fireflies(i,:)) ...
                    + alpha * randn(1,dim).*range;

                if rand < 0.2*(1-progress)
                    LF = LevyFlight(dim);
                    newPosition = newPosition + levy_scale .* LF .* range;
                end

                if rand < 0.15
                    ids = randperm(n,3);
                    F = 0.5*(1-progress) + 0.1;
                    mutant = fireflies(ids(1),:) + F*(fireflies(ids(2),:) - fireflies(ids(3),:));
                    CR = 0.7;
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

    [currentBest, idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

    localStep = 0.02 * range * (1-progress)^3 + 1e-10;

    for k = 1:20
        candidate = BestPosition + localStep .* randn(1,dim);
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

    if stallCounter > 100
        [~, sortedIdx] = sort(fitness,'descend');
        numRestart = round(0.30*n);

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
    ConvergenceCurve(t) = BestScore;

    if abs(BestScore - optimum) <= 1e-12
        ConvergenceCurve(t:end) = BestScore;
        break;
    end
end

[BestScore, BestPosition] = FinalPolishing(BestPosition, BestScore, lb, ub, fobj);
ConvergenceCurve(end) = BestScore;

end


%% =====================================================
% Final Local Search using fminsearch
% =====================================================

function [BestScore, BestPosition] = FinalPolishing(BestPosition, BestScore, lb, ub, fobj)

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


function y = PenalizedObjective(x, lb, ub, fobj)

xClipped = BoundaryCheck(x, lb, ub);
penalty = sum((x - xClipped).^2) * 1e6;

y = fobj(xClipped) + penalty;

end


%% =====================================================
% Levy Flight
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
% Test Functions
% =====================================================

function fobj = GetTestFunction(name, dim)

switch name

    case 'Michalewicz'
        m = 10;
        fobj = @(x) -sum(sin(x) .* (sin(((1:dim).*x.^2)/pi)).^(2*m));

    case 'Ackley'
        fobj = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/dim)) ...
                    - exp(sum(cos(2*pi*x))/dim) + 20 + exp(1);

    case 'Levy1'
        fobj = @(x) LevyFunction(x);

    case 'LevyN13'
        fobj = @(x) sin(3*pi*x(1))^2 ...
                  + (x(1)-1)^2 * (1 + sin(3*pi*x(2))^2) ...
                  + (x(2)-1)^2 * (1 + sin(2*pi*x(2))^2);

    case 'Colville'
        fobj = @(x) 100*(x(1)^2-x(2))^2 + (x(1)-1)^2 ...
                  + (x(3)-1)^2 + 90*(x(3)^2-x(4))^2 ...
                  + 10.1*((x(2)-1)^2 + (x(4)-1)^2) ...
                  + 19.8*(x(2)-1)*(x(4)-1);

    case 'Schwefel'
        fobj = @(x) 418.9828872724338*dim - sum(x .* sin(sqrt(abs(x))));

    case 'Beale'
        fobj = @(x) (1.5 - x(1) + x(1)*x(2))^2 ...
                  + (2.25 - x(1) + x(1)*x(2)^2)^2 ...
                  + (2.625 - x(1) + x(1)*x(2)^3)^2;

    case 'Rosenbrock'
        fobj = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 ...
                    + (x(1:end-1)-1).^2);

    case 'RotatedHyperEllipsoid'
        fobj = @(x) sum(arrayfun(@(i) sum(x(1:i).^2), 1:length(x)));

    otherwise
        error('Unknown test function');
end

end


%% =====================================================
% Levy Function
% =====================================================

function y = LevyFunction(x)

w = 1 + (x - 1)/4;

term1 = sin(pi*w(1))^2;

term2 = 0;
for i = 1:length(x)-1
    term2 = term2 + (w(i)-1)^2 * ...
        (1 + 10*sin(pi*w(i)+1)^2);
end

term3 = (w(end)-1)^2 * ...
    (1 + sin(2*pi*w(end))^2);

y = term1 + term2 + term3;

end