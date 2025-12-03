% This script processes TILDAS .str and .stc files
% This script is used to process data acquired at the University of Cape Town

%% TILDAS BATCH IMPORT AND EXPORT PROCESSED TABLES AS .TXT FILES
clear all; close all; clc;

% Set data folder
dataFolder = '/Users/vincent/Downloads/TeamViewer_ReceivedFiles 18'; % <-- Update this to your path

% Get list of .str and .stc files
strFiles = dir(fullfile(dataFolder, '*.str'));
stcFiles = dir(fullfile(dataFolder, '*.stc'));


strNames = erase({strFiles.name}, '.str');
stcNames = erase({stcFiles.name}, '.stc');

commonNames = intersect(strNames, stcNames);

if isempty(commonNames)
    error('No matching .str and .stc file pairs found.');
end

for iFile = 1:length(commonNames)
    baseFileName = commonNames{iFile};

    %% --- Import .str file ---
    strPath = fullfile(dataFolder, [baseFileName '.str']);
    delimiter = {',',' '};
    formatSpec = '%s%s%s%s%*s%s%*s%s%s%s%[^\n\r]';
    fileID = fopen(strPath,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
        'TextType', 'string', 'ReturnOnError', false, 'HeaderLines', 1);
    fclose(fileID);

    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
    end

    for col=[1,2,3,4,5,6,7,8]
        rawData = dataArray{col};
        for row=1:size(rawData, 1)
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData(row), regexstr, 'names');
                numbers = result.numbers;
                invalidThousandsSeparator = false;
                if numbers.contains(',')
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(numbers, thousandsRegExp, 'once'))
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                if ~invalidThousandsSeparator
                    numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch
                raw{row, col} = rawData{row};
            end
        end
    end
    R = cellfun(@(x) ~isnumeric(x) && ~islogical(x), raw);
    raw(R) = {NaN};

    MoutputTable = table;
    MoutputTable.Datetime = cell2mat(raw(:,1));
    MoutputTable.Xp626_L1 = cell2mat(raw(:,2));
    MoutputTable.Xp627_L1 = cell2mat(raw(:,3));
    MoutputTable.Xp628_L1 = cell2mat(raw(:,4));
    MoutputTable.Xp636_L1 = cell2mat(raw(:,5));
    MoutputTable.Xp628_L2 = cell2mat(raw(:,6));
    MoutputTable.Xp627_L2 = cell2mat(raw(:,7));
    MoutputTable.Xp626_L2 = cell2mat(raw(:,8));
    MoutputTable.dp18O_L2_raw = 1000 * log(MoutputTable.Xp628_L2 ./ MoutputTable.Xp626_L2);
    MoutputTable.dp17O_L2_raw = 1000 * log(MoutputTable.Xp627_L2 ./ MoutputTable.Xp626_L2);
    MoutputTable.dp18O_L1_raw = 1000 * log(MoutputTable.Xp628_L1 ./ MoutputTable.Xp626_L1);
    MoutputTable.dp13C_L1_raw = 1000 * log(MoutputTable.Xp636_L1 ./ MoutputTable.Xp626_L1);

    %% --- Import .stc file ---
    stcPath = fullfile(dataFolder, [baseFileName '.stc']);
    delimiter = ',';
    formatSpec = '%s%*s%*s%*s%*s%*s%*s%s%s%*s%s%s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%s%[^\n\r]';
    %formatSpec = '%s%*s%*s%*s%*s%*s%*s%s%s%s%s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%s%[^\n\r]';;
    fileID = fopen(stcPath,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
        'TextType', 'string', 'ReturnOnError', false, 'HeaderLines', 2);
    fclose(fileID);

    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
    end

    for col=[1,2,3,4,5,6]
        rawData = dataArray{col};
        for row=1:size(rawData, 1)
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData(row), regexstr, 'names');
                numbers = result.numbers;
                invalidThousandsSeparator = false;
                if numbers.contains(',')
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(numbers, thousandsRegExp, 'once'))
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                if ~invalidThousandsSeparator
                    numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch
                raw{row, col} = rawData{row};
            end
        end
    end
    R = cellfun(@(x) ~isnumeric(x) && ~islogical(x), raw);
    raw(R) = {NaN};

    CoutputTable = table;
    CoutputTable.time = cell2mat(raw(:,1));
    CoutputTable.Praw = cell2mat(raw(:,2));
    CoutputTable.Traw = cell2mat(raw(:,5));
    CoutputTable.Pref = cell2mat(raw(:,3));
    CoutputTable.Tref = cell2mat(raw(:,4));
    CoutputTable.ECL_Index = cell2mat(raw(:,6));
    CoutputTable(CoutputTable.ECL_Index == 15,:) = [];

    combinedTable = [MoutputTable, CoutputTable];

    %% --- Processed Table: average per packet ---

    data = combinedTable{:,:};
    nRows = size(data,1);
    output = [];
    currentIndex = data(1,18); %%%%THIS
    Cycle_Index = [];
    packetStart = 1;
    packetCounter11 = 1;
    packetCounter13 = 0;
    skipFirst = true;

    for i = 2:nRows
        if data(i,18) ~= currentIndex || i == nRows
            packetEnd = (i == nRows) * i + (i ~= nRows) * (i-1);
            packet = data(packetStart:packetEnd,:);

            if size(packet,1) > 7
                trimmed = packet(8:end,1:18);
                meanVals = mean(trimmed,1,'omitnan');
            else
                meanVals = nan(1,18);
            end

            packetType = currentIndex;
            if skipFirst
                skipFirst = false;
            else
                if packetType == 11
                    Cycle_Index = 2 * packetCounter11;
                    packetCounter11 = packetCounter11 + 1;
                elseif packetType == 13
                    Cycle_Index = 2 * packetCounter13 + 1;
                    packetCounter13 = packetCounter13 + 1;
                end
                resultRow = [meanVals, packetType, Cycle_Index];
                output = [output; resultRow];
            end

            packetStart = i;
            currentIndex = data(i,18);
        end
    end

    processedTable = array2table(output);
    processedTable(:,19) = [];
    processedTable.Properties.VariableNames = [combinedTable.Properties.VariableNames(1:18), {'Cycle_Index'}];

    %% --- Write processed table to .txt ---
    % --- Manual check/edit of processedTable before writing and summary ---
    % --- keep commented in normal mode.
%   
%     assignin('base', 'processedTable', processedTable);  % Push to workspace
%     disp(['>>> File: ' baseFileName]);
%     disp('>>> "processedTable" has been pushed to the Workspace.');
%     disp('>>> You may now manually edit it.');
%     disp('>>> When ready, type `return` or `dbcont` in the Command Window to resume the script.');
% 
%     keyboard  % Pauses script and drops you into debug mode
    
    %% Output to .txt

    outputTxtPath = fullfile(dataFolder, [baseFileName, '.txt']);
    writetable(processedTable, outputTxtPath, 'Delimiter', '\t');

    [cleanMismatch, flaggedCycles] = calculateMismatch(processedTable);
    sampleCycles = processedTable.Cycle_Index(processedTable.ECL_Index==11);
    first3       = unique(sampleCycles);
    first3       = first3(1:min(3,numel(first3)));  
    maskFirst3   = ismember(processedTable.Cycle_Index, first3) ...
                   & processedTable.ECL_Index==11;
    processedTable(maskFirst3, :) = [];

    % 2) Then drop any remaining sample cycles flagged as outliers
    maskOutliers = ismember(processedTable.Cycle_Index, flaggedCycles) ...
                   & processedTable.ECL_Index==11;
    processedTable(maskOutliers, :) = [];

    % ?????????????????????????????????????????????????????????????
    % Build one?row summary for this file
    % ?????????????????????????????????????????????????????????????

    % 2) Sample?only subset of processedTable
    ps = processedTable(processedTable.ECL_Index==11,:);

    % 3) Means & stds on processedTable
    m_Xp626   = mean(ps.Xp626_L2,'omitnan');   s_Xp626   = std(ps.Xp626_L2,'omitnan');
    m_Praw    = mean(ps.Praw,      'omitnan'); s_Praw    = std(ps.Praw,      'omitnan');
    m_Traw    = mean(ps.Traw,      'omitnan'); s_Traw    = std(ps.Traw,      'omitnan');

    % 4) Means & stds on cleanMismatch
    mm_Xp626  = mean(cleanMismatch.Xp626_L2,'omitnan'); sms_Xp626 = std(cleanMismatch.Xp626_L2,'omitnan');
    mm_Praw   = mean(cleanMismatch.Praw,     'omitnan'); sms_Praw  = std(cleanMismatch.Praw,     'omitnan');
    mm_Traw   = mean(cleanMismatch.Traw,     'omitnan'); sms_Traw  = std(cleanMismatch.Traw,     'omitnan');

    % 5) Counts
    sample_n  = height(cleanMismatch);
    outliers  = numel(flaggedCycles);

    % 6) Additional dp?columns on cleanMismatch
    m_dp13    = mean(cleanMismatch.dp13C_L1_raw, 'omitnan'); s_dp13   = std(cleanMismatch.dp13C_L1_raw, 'omitnan');
    m_dp18_L1 = mean(cleanMismatch.dp18O_L1_raw, 'omitnan'); s_dp18_L1= std(cleanMismatch.dp18O_L1_raw, 'omitnan');
    m_dp18_L2 = mean(cleanMismatch.dp18O_L2_raw, 'omitnan'); s_dp18_L2= std(cleanMismatch.dp18O_L2_raw, 'omitnan');
    m_Dp17    = mean(cleanMismatch.Dp17O_raw,   'omitnan'); s_Dp17   = std(cleanMismatch.Dp17O_raw,   'omitnan');

    mm_dp17L2 = mean(cleanMismatch.dp17O_L2_raw, 'omitnan'); sms_dp17L2 = std(cleanMismatch.dp17O_L2_raw, 'omitnan');

    % 7) Assemble into one-row table
    SummaryOne = table( ...
      {baseFileName}, ...
      m_Xp626,   s_Xp626,   mm_Xp626,   sms_Xp626, ...
      m_Praw,    s_Praw,    mm_Praw,    sms_Praw,  ...
      m_Traw,    s_Traw,    mm_Traw,    sms_Traw,  ...
      sample_n,  outliers,  ...
      m_dp13,    s_dp13,    ...
      m_dp18_L1, s_dp18_L1, ...
      m_dp18_L2, s_dp18_L2, ...
      mm_dp17L2, sms_dp17L2, ...   % <-- new columns here
      m_Dp17,    s_Dp17, ...
      'VariableNames', { ...
        'Name', ...
        'Xp626_L2_mean','Xp626_L2_std','mismatch_Xp626_L2','std_mismatch_Xp626_L2', ...
        'Praw_mean','Praw_std','mismatch_Praw','std_mismatch_Praw', ...
        'Traw_mean','Traw_std','mismatch_Traw','std_mismatch_Traw', ...
        'sample_n','outliers', ...
        'dp13C_L1_raw_mean','dp13C_L1_raw_std', ...
        'dp18O_L1_raw_mean','dp18O_L1_raw_std', ...
        'dp18O_L2_raw_mean','dp18O_L2_raw_std', ...
        'dp17O_L2_raw_mean','dp17O_L2_raw_std', ...  % <-- new names
        'Dp17O_raw_mean','Dp17O_raw_std' } );

    % 2. Build your reduced summary filename
    reducedFileName   = ['Reduced_' baseFileName '.txt'];

    % 3. Construct the full path to the new .txt in the subfolder
    summaryFilePath   = fullfile(dataFolder, reducedFileName);

    % 4. Write the one-row summary table
    writetable(SummaryOne, summaryFilePath, 'Delimiter', '\t');
    
end

summaryFolder = dataFolder% '/Users/vincent/Downloads/TeamViewer_ReceivedFiles';  % adjust if needed

% List all Reduced_*.txt files
reducedFiles = dir(fullfile(summaryFolder, 'Reduced_*.txt'));

% Preallocate
allSummaries = {};

for i = 1:length(reducedFiles)
    filePath = fullfile(summaryFolder, reducedFiles(i).name);
    
    % Read the one-row summary table
    T = readtable(filePath, 'Delimiter', '\t');

    % Extract timestamp string from filename
    [~, baseName, ~] = fileparts(reducedFiles(i).name);  % e.g. 'Reduced_241007_164546'
    timestampStr = extractAfter(baseName, 'Reduced_');    % e.g. '241007_164546'

    % Convert to datetime
    try
        dt = datetime(timestampStr, 'InputFormat', 'yyMMdd_HHmmss');
    catch
        warning('Skipping file due to invalid timestamp: %s', reducedFiles(i).name);
        continue
    end

    % Add formatted date and time
    T.Date = string(datestr(dt, 'yyyy mmm dd'));
    T.Time = string(datestr(dt, 'HH:MM:SS'));

    allSummaries{end+1} = T;
end

% Combine all into one table
combinedSummary = vertcat(allSummaries{:});

% Reorder columns: Date, Time first
otherVars = setdiff(combinedSummary.Properties.VariableNames, {'Date','Time'}, 'stable');
combinedSummary = combinedSummary(:, [{'Date','Time'}, otherVars]);

% Write to CSV
outputCSV = fullfile(summaryFolder, 'Combined_Summary.csv');
writetable(combinedSummary, outputCSV);
fprintf('? Combined summary saved to: %s\n', outputCSV);

function [mismatchTable, outlierCycles] = calculateMismatch(processedTable)
% calculateMismatch  Compute sample?reference mismatches, drop first 4 cycles,
%                   add Dp17O_raw, flag & remove outliers.
%
%   [mismatchTable, outlierCycles] = calculateMismatch(processedTable)
%
% Inputs:
%   processedTable   ? table from Script 1 with numeric columns, plus
%                      ECL_Index (11=sample,13=reference) and Cycle_Index.
%
% Outputs:
%   mismatchTable    ? table of mismatches for sample cycles (post?filter), 
%                      with added Dp17O_raw, outliers removed
%   outlierCycles    ? vector of Cycle_Index values flagged as outliers

  % 1) Split out sample and reference rows
  samples = processedTable(processedTable.ECL_Index==11, :);
  refs    = processedTable(processedTable.ECL_Index==13, :);

  % 2) Drop the first 3 sample cycles (by Cycle_Index)
  uc = unique(samples.Cycle_Index);
  if numel(uc) < 3
    mismatchTable = table(); 
    outlierCycles = [];
    return;
  end
  cutoff = uc(3);
  samples(samples.Cycle_Index <= cutoff, :) = [];

  % 3) Prepare to compute mismatches
  allNames = processedTable.Properties.VariableNames;
  numCols  = setdiff(allNames, {'ECL_Index','Cycle_Index'});
  nS = height(samples);
  nV = numel(numCols);
  dataOut = nan(nS, nV+2);

  % 4) Loop over remaining sample cycles
  for i = 1:nS
    c = samples.Cycle_Index(i);
    idxB = find(refs.Cycle_Index < c,  1, 'last');
    idxA = find(refs.Cycle_Index > c,  1, 'first');
    if isempty(idxB)||isempty(idxA)
      continue
    end
    refMean  = (refs{idxB, numCols} + refs{idxA, numCols})/2;
    sampVals = samples{i,      numCols};
    mismatch = sampVals - refMean;
    dataOut(i,:) = [mismatch, samples.ECL_Index(i), samples.Cycle_Index(i)];
  end

  % 5) Build the mismatch table
  mismatchTable = array2table(dataOut, ...
      'VariableNames',[numCols, {'ECL_Index','Cycle_Index'}]);

  % 6) Compute Dp17O_raw and flag outliers
  mismatchTable.Dp17O_raw = mismatchTable.dp17O_L2_raw ...
                            - 0.528*mismatchTable.dp18O_L2_raw;
  isOut = isoutlier(mismatchTable.Dp17O_raw);
  outlierCycles = mismatchTable.Cycle_Index(isOut);

  % 7) Remove the outlier rows
  mismatchTable = mismatchTable(~isOut, :);
end

function resume_script
    disp('Resuming script...');
    dbcont
end
