%% Step 00: Build a sanitized CT-series manifest from Labeler sessions.
% Set sessionRoot, dicomRoot, manifestPath, patientIds, or expectedPatients
% before run(...) to override the portable defaults.

codeRoot = fileparts(fileparts(mfilename('fullpath')));
projectRoot = fileparts(codeRoot);
dataRoot = fileparts(projectRoot);
if ~exist('sessionRoot', 'var') || isempty(sessionRoot)
    sessionRoot = fullfile(dataRoot, 'SAVED SESSIONS');
end
if ~exist('dicomRoot', 'var') || isempty(dicomRoot)
    dicomRoot = fullfile(dataRoot, 'DICOM');
end
if ~exist('manifestPath', 'var') || isempty(manifestPath)
    manifestPath = fullfile(codeRoot, 'config', ...
        'session_series_manifest.csv');
end
if ~exist('expectedPatients', 'var')
    expectedPatients = [];
end
if ~exist('patientIds', 'var')
    patientIds = [];
end
sessionRoot = char(sessionRoot);
dicomRoot = char(dicomRoot);
manifestPath = char(manifestPath);

if ~isfolder(sessionRoot)
    error('Manifest:MissingSessions', 'Missing session folder: %s', sessionRoot);
end
if ~isfolder(dicomRoot)
    error('Manifest:MissingDicom', 'Missing DICOM folder: %s', dicomRoot);
end

sessionDirs = dir(fullfile(sessionRoot, 'PAE*'));
sessionDirs = sessionDirs([sessionDirs.isdir]);
[sessionDirs, ~] = sortPatientDirectories(sessionDirs);
if ~isempty(patientIds)
    requestedV = upper(string(patientIds(:)));
    availableV = upper(string({sessionDirs.name}));
    missingV = setdiff(requestedV, availableV);
    if ~isempty(missingV)
        error('Manifest:UnknownPatient', 'Unknown patient: %s.', missingV(1));
    end
    sessionDirs = sessionDirs(ismember(availableV, requestedV));
end
if isempty(sessionDirs)
    error('Manifest:NoPatients', 'No PAE session folders found.');
end

nPatients = numel(sessionDirs);
patientIdV = strings(nPatients, 1);
seriesUidV = strings(nPatients, 1);
seriesFolderV = strings(nPatients, 1);
sourceFileCountV = nan(nPatients, 1);
statusV = strings(nPatients, 1);

for index = 1:nPatients
    patientId = sessionDirs(index).name;
    patientIdV(index) = string(patientId);
    sessionPath = fullfile(sessionDirs(index).folder, patientId, ...
        'AppData', 'Session.mat');
    fprintf('[%d/%d] %s\n', index, nPatients, patientId);

    try
        if exist(sessionPath, 'file') ~= 2
            error('Manifest:MissingSession', 'Missing %s.', sessionPath);
        end
        loadedS = load(sessionPath, 'sessionInfo');
        if ~isfield(loadedS, 'sessionInfo')
            error('Manifest:MissingSessionInfo', ...
                'sessionInfo is missing from %s.', sessionPath);
        end

        sourcePaths = collectDataSources(loadedS.sessionInfo.DataEntries);
        if isempty(sourcePaths)
            error('Manifest:NoDataSource', ...
                'No DICOM DataSource paths found for %s.', patientId);
        end

        folders = strings(numel(sourcePaths), 1);
        for pathIndex = 1:numel(sourcePaths)
            folders(pathIndex) = seriesFolderFromPath( ...
                sourcePaths(pathIndex), patientId);
        end
        uniqueFolders = unique(folders);
        if numel(uniqueFolders) ~= 1
            error('Manifest:MultipleSeries', ...
                'Session.mat references %d series folders for %s.', ...
                numel(uniqueFolders), patientId);
        end

        seriesFolder = char(uniqueFolders(1));
        seriesPath = fullfile(dicomRoot, patientId, seriesFolder);
        if ~isfolder(seriesPath)
            error('Manifest:SeriesNotFound', ...
                'DICOM series folder not found: %s', seriesPath);
        end
        serverFiles = dir(fullfile(seriesPath, '**', '*.dcm'));
        serverFiles = serverFiles(~[serverFiles.isdir]);
        if numel(serverFiles) ~= numel(sourcePaths)
            error('Manifest:FileCount', ...
                'Session/server DICOM counts differ for %s: %d versus %d.', ...
                patientId, numel(sourcePaths), numel(serverFiles));
        end

        seriesUidV(index) = string(strrep(seriesFolder, '_', '.'));
        seriesFolderV(index) = string(seriesFolder);
        sourceFileCountV(index) = numel(sourcePaths);
        statusV(index) = string('ok');
    catch ME
        statusV(index) = string(['error: ' ME.message]);
    end
end

manifestT = table(patientIdV, seriesUidV, seriesFolderV, ...
    sourceFileCountV, statusV, 'VariableNames', ...
    {'patient_id', 'series_uid', 'series_folder', ...
    'source_file_count', 'status'});
manifestFolder = fileparts(manifestPath);
if ~isfolder(manifestFolder)
    mkdir(manifestFolder);
end
writetable(manifestT, manifestPath);
fprintf('Manifest written: %s\n', manifestPath);

failedV = ~strcmp(statusV, 'ok');
if any(failedV)
    error('Manifest:Incomplete', '%d of %d session rows failed.', ...
        nnz(failedV), nPatients);
end
if ~isempty(expectedPatients) && nPatients ~= expectedPatients
    error('Manifest:PatientCount', 'Expected %d patients, found %d.', ...
        expectedPatients, nPatients);
end
fprintf('Complete: %d validated patient-series rows.\n', nPatients);


function [directories, numbers] = sortPatientDirectories(directories)
% Keep valid PAE identifiers and sort them numerically.
names = {directories.name};
tokens = regexp(names, '^PAE(\d+)$', 'tokens', 'once', 'ignorecase');
validV = ~cellfun(@isempty, tokens);
directories = directories(validV);
tokens = tokens(validV);
numbers = cellfun(@(value) str2double(value{1}), tokens);
[numbers, orderV] = sort(numbers);
directories = directories(orderV);
end


function paths = collectDataSources(entries)
% Read only sessionInfo.DataEntries.DataSource.
if isstruct(entries)
    if ~isfield(entries, 'DataSource')
        error('Manifest:MissingDataSource', 'DataEntries has no DataSource field.');
    end
    values = {entries.DataSource};
elseif istable(entries)
    if ~ismember('DataSource', entries.Properties.VariableNames)
        error('Manifest:MissingDataSource', 'DataEntries has no DataSource column.');
    end
    values = {entries.DataSource};
elseif isobject(entries) && isprop(entries, 'DataSource')
    values = {entries.DataSource};
else
    error('Manifest:UnsupportedEntries', ...
        'Unsupported DataEntries type: %s.', class(entries));
end

paths = strings(0, 1);
for index = 1:numel(values)
    paths = [paths; flattenStrings(values{index})]; %#ok<AGROW>
end
paths = strtrim(paths);
paths = paths(endsWith(lower(paths), '.dcm'));
paths = unique(paths, 'stable');
end


function output = flattenStrings(value)
% Convert common DataSource containers to one string column.
if isstring(value)
    output = value(:);
elseif ischar(value)
    if isrow(value)
        output = string(value);
    else
        output = string(cellstr(value));
    end
elseif iscell(value)
    output = strings(0, 1);
    for index = 1:numel(value)
        output = [output; flattenStrings(value{index})]; %#ok<AGROW>
    end
else
    error('Manifest:UnsupportedDataSource', ...
        'Unsupported DataSource value type: %s.', class(value));
end
end


function folder = seriesFolderFromPath(sourcePath, patientId)
% The selected series folder immediately follows the patient ID in DataSource.
parts = regexp(char(sourcePath), '[\\/]+', 'split');
patientIndex = find(strcmpi(parts, patientId), 1, 'first');
if isempty(patientIndex) || patientIndex >= numel(parts)
    error('Manifest:InvalidDataSource', ...
        'Cannot identify the series folder in %s.', char(sourcePath));
end
folder = string(parts{patientIndex + 1});
if strlength(folder) == 0
    error('Manifest:InvalidDataSource', 'The series folder is empty.');
end
end
