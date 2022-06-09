SpikeTimes = {unit.SpikeTimes};
Trial_idx_of_spike = {unit.Trial_idx_of_spike};
Trial_info = {unit.Trial_info};
%%

Rtot = zeros(length(unit),1);
F_v = zeros(length(unit),1);
t_viol = 0.0025;

for i = 1:length(unit)

    range = Trial_info{i}.Trial_range_to_analyze;
    num_trials = range(2) - range(1) + 1;
    T = num_trials*6; 

    Rtot(i) = length(SpikeTimes{i})/T;

    ISI_viols = 0;
    for j = range(1):range(2)
        indices = Trial_idx_of_spike{i} == j;
        trial = SpikeTimes{i}(indices);
        ISI_viols = ISI_viols + sum(diff(trial) < t_viol);
    end

    F_v(i) = ISI_viols/length(SpikeTimes{i});

end
%%

times = [];
for i = 1:length(unit)
    times = vertcat(times,SpikeTimes{i});
end

disp(min(times))
disp(max(times))

%% asdf
myFolder = 'C:\Users\jpv88\OneDrive\Documents\GitHub\Chand-Lab\SiliconProbeData\FixedDelayTask\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

Rtot_full = [];
F_v_full = [];
for i = 1:length(names)
    unit = load(strcat(myFolder,names{i})).unit;
    [Rtot,F_v] = extract_Rtot_Fv(unit);
    Rtot_full = vertcat(Rtot_full,Rtot);
    F_v_full = vertcat(F_v_full,F_v);
end

writematrix(Rtot_full,'Rtot.csv')
writematrix(F_v_full,'F_v.csv')

%% asdf
myFolder = 'C:\Users\jpv88\OneDrive\Documents\GitHub\Chand-Lab\SiliconProbeData\FixedDelayTask\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

spikes_full = {};
trials_full = {};
ranges_full = [];
session_full = [];
for i = 1:length(names)
    unit = load(strcat(myFolder,names{i})).unit;
    session_full = vertcat(session_full,repelem(i-1,length(unit))');
    [spikes,trials,ranges] = extract_times(unit);
    spikes_full = vertcat(spikes_full,spikes');
    trials_full = vertcat(trials_full,trials');
    ranges_full = vertcat(ranges_full,ranges);
end

writecell(spikes_full,'spikes.csv')
writecell(trials_full,'trials.csv')
writematrix(ranges_full,'ranges.csv')
writematrix(session_full,'sessions.csv')
%% Functions

function [Rtot,F_v] = extract_Rtot_Fv(unit)

    SpikeTimes = {unit.SpikeTimes};
    Trial_idx_of_spike = {unit.Trial_idx_of_spike};
    Trial_info = {unit.Trial_info};

    Rtot = zeros(length(unit),1);
    F_v = zeros(length(unit),1);
    t_viol = 0.0025;
    
    for i = 1:length(unit)
    
        range = Trial_info{i}.Trial_range_to_analyze;
        num_trials = range(2) - range(1) + 1;
        T = num_trials*6; 
    
        Rtot(i) = length(SpikeTimes{i})/T;
    
        ISI_viols = 0;
        for j = range(1):range(2)
            indices = Trial_idx_of_spike{i} == j;
            trial = SpikeTimes{i}(indices);
            ISI_viols = ISI_viols + sum(diff(trial) < t_viol);
        end
    
        F_v(i) = ISI_viols/length(SpikeTimes{i});
    
    end
end

% extract trial range to analyze, spike times, and trial_idx for a given
% session
function [spikes, trials, ranges] = extract_times(unit)
    SpikeTimes = {unit.SpikeTimes};
    Trial_idx_of_spike = {unit.Trial_idx_of_spike};
    Trial_info = {unit.Trial_info};
    ranges = zeros(length(unit),2);
    spikes = SpikeTimes;
    trials = Trial_idx_of_spike;

    for i = 1:length(unit)
        range = Trial_info{i}.Trial_range_to_analyze;
        ranges(i,:) = range;
    end
    
end


