clc;
clear;
close all;

%% =====================================================
% STATISTICAL BENCHMARKING OF FA, RAC-FA, PSO, GWO, SMO
% =====================================================

Runs = 10;
SearchAgents = 30;
MaxIter = 500;

Algorithms = {'FA','RAC-FA','PSO','GWO','SMO'};

Benchmarks = {
    'Sphere',       -100,     100,     30, 0, 1e-5;
    'Rosenbrock',    -30,      30,     30, 0, 1e-1;
    'Rastrigin',    -5.12,   5.12,     30, 0, 1e-5;
    'Ackley',      -32.768, 32.768,    30, 0, 1e-5;
    'Griewank',     -600,     600,     30, 0, 1e-5;
    'Schwefel',     -500,     500,     30, 0, 1e-5;
    'Zakharov',       -5,      10,     30, 0, 1e-5;
    'Michalewicz',     0,      pi,     10, -9.66015, 1e-5;
};

numFunctions = size(Benchmarks,1);
numAlgorithms = length(Algorithms);

AllResults = cell(numFunctions,1);
AverageCurves = cell(numFunctions,1);

MeanFitnessTableData = zeros(numFunctions,numAlgorithms);
MeanErrorTableData   = zeros(numFunctions,numAlgorithms);
StdErrorTableData    = zeros(numFunctions,numAlgorithms);
MeanRuntimeTableData = zeros(numFunctions,numAlgorithms);

BestAlgorithm = strings(numFunctions,1);
BestMeanError = zeros(numFunctions,1);

%% =====================================================
% MAIN LOOP
% =====================================================

for f = 1:numFunctions

    fname = Benchmarks{f,1};
    lb = Benchmarks{f,2};
    ub = Benchmarks{f,3};
    dim = Benchmarks{f,4};
    optimum = Benchmarks{f,5};
    errorLimit = Benchmarks{f,6};

    fobj = GetBenchmarkFunction(fname, dim);

    fprintf('\n============================================\n');
    fprintf('Benchmark Function: %s\n', fname);
    fprintf('Dimension: %d\n', dim);
    fprintf('Search Range: [%g, %g]\n', lb, ub);
    fprintf('Optimum Value: %.10f\n', optimum);
    fprintf('============================================\n');

    BestValues = zeros(Runs,numAlgorithms);
    RuntimeValues = zeros(Runs,numAlgorithms);
    Curves = zeros(MaxIter,numAlgorithms,Runs);

    for r = 1:Runs

        fprintf('Run %d/%d...\n', r, Runs);

        %% ---------------- FA ----------------
        tic;
        [best_FA, curve_FA] = StandardFA(SearchAgents, MaxIter, lb, ub, dim, fobj);
        RuntimeValues(r,1) = toc;
        BestValues(r,1) = best_FA;
        Curves(:,1,r) = curve_FA;

        %% ---------------- RAC-FA ----------------
        tic;
        [best_RACFA, ~, curve_RACFA] = RAC_FA(SearchAgents, MaxIter, lb, ub, dim, fobj, optimum);
        RuntimeValues(r,2) = toc;
        BestValues(r,2) = best_RACFA;
        Curves(:,2,r) = curve_RACFA;

        %% ---------------- PSO ----------------
        tic;
        [best_PSO, curve_PSO] = PSO(SearchAgents, MaxIter, lb, ub, dim, fobj);
        RuntimeValues(r,3) = toc;
        BestValues(r,3) = best_PSO;
        Curves(:,3,r) = curve_PSO;

        %% ---------------- GWO ----------------
        tic;
        [best_GWO, curve_GWO] = GWO(SearchAgents, MaxIter, lb, ub, dim, fobj);
        RuntimeValues(r,4) = toc;
        BestValues(r,4) = best_GWO;
        Curves(:,4,r) = curve_GWO;

        %% ---------------- SMO ----------------
        tic;
        [best_SMO, curve_SMO] = SMO(SearchAgents, MaxIter, lb, ub, dim, fobj);
        RuntimeValues(r,5) = toc;
        BestValues(r,5) = best_SMO;
        Curves(:,5,r) = curve_SMO;

    end

    %% =====================================================
    % STATISTICAL ANALYSIS
    % =====================================================

    ErrorValues = abs(BestValues - optimum);

    BestError = min(ErrorValues)';
    WorstError = max(ErrorValues)';
    MeanError = mean(ErrorValues)';
    StdError = std(ErrorValues)';
    MeanFitness = mean(BestValues)';
    MeanRuntime = mean(RuntimeValues)';

    SuccessRate = zeros(numAlgorithms,1);

    for a = 1:numAlgorithms
        SuccessRate(a) = sum(ErrorValues(:,a) <= errorLimit) / Runs * 100;
    end

    ResultTable = table( ...
        Algorithms', ...
        MeanFitness, ...
        BestError, ...
        WorstError, ...
        MeanError, ...
        StdError, ...
        MeanRuntime, ...
        SuccessRate, ...
        'VariableNames', {'Algorithm','Mean_Fitness','Best_Error','Worst_Error','Mean_Error','Std_Error','Mean_Runtime_sec','Success_Rate_percent'});

    AllResults{f} = ResultTable;

    MeanFitnessTableData(f,:) = MeanFitness';
    MeanErrorTableData(f,:) = MeanError';
    StdErrorTableData(f,:) = StdError';
    MeanRuntimeTableData(f,:) = MeanRuntime';

    [BestMeanError(f), bestIdx] = min(MeanError);
    BestAlgorithm(f) = Algorithms{bestIdx};

    disp(' ');
    fprintf('============ RESULTS FOR %s ============\n', fname);
    disp(ResultTable);

    %% =====================================================
    % GROUPED PLOTS
    % =====================================================

    AvgCurve = mean(Curves,3);
    AverageCurves{f} = AvgCurve;
    AvgErrorCurve = abs(AvgCurve - optimum);

    figure('Name',['Summary Plots - ',fname], ...
        'NumberTitle','off', ...
        'Position',[100 50 1400 700]);

    subplot(2,2,1)
    plot(AvgCurve(:,1),'r','LineWidth',2); hold on;
    plot(AvgCurve(:,2),'b','LineWidth',2);
    plot(AvgCurve(:,3),'g','LineWidth',2);
    plot(AvgCurve(:,4),'m','LineWidth',2);
    plot(AvgCurve(:,5),'k','LineWidth',2);
    grid on;
    xlabel('Iteration');
    ylabel('Mean Best Fitness');
    title(['Fitness Curve - ',fname]);
    legend(Algorithms,'Location','best');

    subplot(2,2,2)
    semilogy(max(AvgErrorCurve(:,1),eps),'r','LineWidth',2); hold on;
    semilogy(max(AvgErrorCurve(:,2),eps),'b','LineWidth',2);
    semilogy(max(AvgErrorCurve(:,3),eps),'g','LineWidth',2);
    semilogy(max(AvgErrorCurve(:,4),eps),'m','LineWidth',2);
    semilogy(max(AvgErrorCurve(:,5),eps),'k','LineWidth',2);
    grid on;
    xlabel('Iteration');
    ylabel('Mean Error');
    title(['Error Convergence - ',fname]);
    legend(Algorithms,'Location','best');

    subplot(2,2,3)
    bar(MeanFitness);
    grid on;
    set(gca,'XTickLabel',Algorithms);
    xlabel('Algorithm');
    ylabel('Mean Fitness');
    title(['Mean Fitness Comparison - ',fname]);

    subplot(2,2,4)
    bar(MeanError);
    grid on;
    set(gca,'XTickLabel',Algorithms);
    xlabel('Algorithm');
    ylabel('Mean Error');
    title(['Mean Error Comparison - ',fname]);

    figure('Name',['Error Distribution - ',fname], ...
        'NumberTitle','off', ...
        'Position',[200 100 900 500]);

    boxplot(ErrorValues, Algorithms);
    grid on;
    ylabel('Final Error');
    title(['Error Distribution of Algorithms on ',fname]);

end

%% =====================================================
% FINAL COMPARISON TABLES
% =====================================================

FunctionNames = string(Benchmarks(:,1));

MeanFitnessComparison = array2table(MeanFitnessTableData, ...
    'VariableNames', Algorithms, ...
    'RowNames', cellstr(FunctionNames));

MeanErrorComparison = array2table(MeanErrorTableData, ...
    'VariableNames', Algorithms, ...
    'RowNames', cellstr(FunctionNames));

StdErrorComparison = array2table(StdErrorTableData, ...
    'VariableNames', Algorithms, ...
    'RowNames', cellstr(FunctionNames));

RuntimeComparison = array2table(MeanRuntimeTableData, ...
    'VariableNames', Algorithms, ...
    'RowNames', cellstr(FunctionNames));

ComparisonTable = table( ...
    FunctionNames, ...
    BestAlgorithm, ...
    BestMeanError, ...
    'VariableNames', {'Function','Best_Algorithm','Lowest_Mean_Error'});

disp(' ');
disp('==============================================');
disp('MEAN FITNESS COMPARISON TABLE');
disp('==============================================');
disp(MeanFitnessComparison);

disp(' ');
disp('==============================================');
disp('MEAN ERROR COMPARISON TABLE');
disp('==============================================');
disp(MeanErrorComparison);

disp(' ');
disp('==============================================');
disp('STANDARD DEVIATION COMPARISON TABLE');
disp('==============================================');
disp(StdErrorComparison);

disp(' ');
disp('==============================================');
disp('MEAN RUNTIME COMPARISON TABLE');
disp('==============================================');
disp(RuntimeComparison);

disp(' ');
disp('==============================================');
disp('BEST ALGORITHM COMPARISON TABLE');
disp('==============================================');
disp(ComparisonTable);

%% =====================================================
% OVERALL GROUPED BAR CHARTS
% =====================================================

figure('Name','Overall Mean Fitness Comparison','NumberTitle','off');
bar(MeanFitnessTableData);
grid on;
xlabel('Benchmark Function');
ylabel('Mean Fitness');
title('Overall Mean Fitness Comparison');
set(gca,'XTickLabel',FunctionNames);
legend(Algorithms,'Location','best');

figure('Name','Overall Mean Error Comparison','NumberTitle','off');
bar(log10(MeanErrorTableData + 1));
grid on;
xlabel('Benchmark Function');
ylabel('log_{10}(Mean Error + 1)');
title('Overall Mean Error Comparison');
set(gca,'XTickLabel',FunctionNames);
legend(Algorithms,'Location','best');

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

BestScore = min(fitness);
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

    currentBest = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
    end

    Curve(t) = BestScore;

end

end

%% =====================================================
% PSO
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
% GWO
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
% SMO
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
                    + gbest_weight * rand(1,dim) .* (BestPosition - fireflies(i,:)) ...
                    + alpha * randn(1,dim) .* range;

                if rand < 0.2 * (1 - progress)
                    LF = LevyFlight(dim);
                    newPosition = newPosition + levy_scale .* LF .* range;
                end

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

    [currentBest, idx] = min(fitness);

    if currentBest < BestScore
        BestScore = currentBest;
        BestPosition = fireflies(idx,:);
    end

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

    if abs(previousBest - BestScore) < 1e-12
        stallCounter = stallCounter + 1;
    else
        stallCounter = 0;
    end

    if stallCounter > stagnationLimit

        [~, sortedIdx] = sort(fitness,'descend');
        numRestart = round(restartFraction * n);

        for rr = 1:numRestart

            id = sortedIdx(rr);

            if rand < 0.5
                fireflies(id,:) = lb + rand(1,dim) .* range;
            else
                fireflies(id,:) = BestPosition + 0.2 * range .* randn(1,dim);
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

    if nargin >= 7 && ~isempty(optimum)
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
% BENCHMARK FUNCTIONS
% =====================================================

function fobj = GetBenchmarkFunction(fname, dim)

switch lower(fname)

    case 'sphere'
        fobj = @(x) sum(x.^2);

    case 'rosenbrock'
        fobj = @(x) sum(100*(x(2:end)-x(1:end-1).^2).^2 ...
            + (x(1:end-1)-1).^2);

    case 'rastrigin'
        fobj = @(x) sum(x.^2 - 10*cos(2*pi*x) + 10);

    case 'ackley'
        fobj = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/dim)) ...
            - exp(sum(cos(2*pi*x))/dim) ...
            + 20 + exp(1);

    case 'griewank'
        fobj = @(x) 1 + sum(x.^2)/4000 ...
            - prod(cos(x ./ sqrt(1:dim)));

    case 'schwefel'
        fobj = @(x) 418.9829*dim ...
            - sum(x .* sin(sqrt(abs(x))));

    case 'zakharov'
        ii = 1:dim;
        fobj = @(x) sum(x.^2) ...
            + (sum(0.5*ii.*x))^2 ...
            + (sum(0.5*ii.*x))^4;

    case 'michalewicz'
        m = 10;
        ii = 1:dim;
        fobj = @(x) -sum(sin(x) .* ...
            (sin(ii .* x.^2 ./ pi)).^(2*m));

    otherwise
        error('Unknown benchmark function');

end

end

%% =====================================================
% PENALIZED OBJECTIVE
% =====================================================

function y = PenalizedObjective(x, lb, ub, fobj)

xClipped = BoundaryCheck(x, lb, ub);
penalty = sum((x - xClipped).^2)*1e6;

y = fobj(xClipped) + penalty;

end

%% =====================================================
% FINAL LOCAL POLISHING
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

%% =====================================================
% LEVY FLIGHT
% =====================================================

function LF = LevyFlight(dim)

beta = 1.5;

sigma = ...
    (gamma(1+beta)*sin(pi*beta/2) / ...
    (gamma((1+beta)/2)*beta ...
    * 2^((beta-1)/2)))^(1/beta);

u = randn(1,dim)*sigma;
v = randn(1,dim);

LF = u ./ abs(v).^(1/beta);

end

%% =====================================================
% BOUNDARY CHECK
% =====================================================

function x = BoundaryCheck(x, lb, ub)

x = max(x, lb);
x = min(x, ub);

end