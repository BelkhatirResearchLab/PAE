%% Step 04: Extract the complete Original CERR radiomics feature set.
% Inputs are validated 3D CT-mask MAT files resampled to 1 mm isotropic.
% The saved featureS and flatS include all configured families, including GLDM.

% Set these variables before run(...) to override the portable defaults.
codeRoot = fileparts(fileparts(mfilename('fullpath')));
projectRoot = fileparts(codeRoot);
if ~exist('paramPath', 'var') || isempty(paramPath)
    paramPath = fullfile(codeRoot, 'config', ...
        'cerr_core_radiomics_params_full_original.json');
end
if ~exist('inputDir', 'var') || isempty(inputDir)
    inputDir = fullfile(projectRoot, 'outputs_final', 'radiomics_inputs_mat');
end
if ~exist('outputDir', 'var') || isempty(outputDir)
    outputDir = fullfile(projectRoot, 'outputs_final', 'radiomics_full');
end
if ~exist('expectedFeatures', 'var') || isempty(expectedFeatures)
    expectedFeatures = 119;
end
if ~exist('expectedPatients', 'var')
    expectedPatients = [];
end
paramPath = char(paramPath);
inputDir = char(inputDir);
outputDir = char(outputDir);
patientDir = fullfile(outputDir, 'per_patient_features');
featureCsv = fullfile(outputDir, 'radiomics_features_CERR_full.csv');
logCsv = fullfile(outputDir, 'radiomics_extraction_log.csv');

if exist(paramPath, 'file') ~= 2
    error('CERR:MissingConfig', 'Missing parameter file: %s', paramPath);
end
if ~isfolder(inputDir)
    error('CERR:MissingInputs', 'Missing input folder: %s', inputDir);
end

addCerrToPath(projectRoot);
rawParamS = jsondecode(fileread(paramPath));
if ~strcmpi(rawParamS.settings.texture.directionality, '3D')
    error('CERR:Not3D', 'The radiomics configuration must use 3D texture directions.');
end
paramS = getRadiomicsParamTemplate(paramPath);
makeDirectory(outputDir);
makeDirectory(patientDir);

matFiles = dir(fullfile(inputDir, 'PAE*.mat'));
matFiles = filterMatFiles(matFiles);
if isempty(matFiles)
    error('CERR:NoInputs', 'No PAE*.mat inputs found in %s', inputDir);
end
if ~isempty(expectedPatients) && numel(matFiles) ~= expectedPatients
    error('CERR:PatientCount', 'Expected %d patients, found %d.', ...
        expectedPatients, numel(matFiles));
end

featureRows = {};
logRows = {};
referenceFeatureNames = {};
for index = 1:numel(matFiles)
    matPath = fullfile(matFiles(index).folder, matFiles(index).name);
    [~, patientId] = fileparts(matPath);
    fprintf('[%d/%d] %s\n', index, numel(matFiles), patientId);

    try
        inputS = load(matPath, 'ct_hu', 'mask', 'metadata_json');
        scan3M = permute(double(inputS.ct_hu), [3 2 1]); % Z,Y,X -> X,Y,Z
        mask3M = permute(logical(inputS.mask), [3 2 1]);
        if ndims(scan3M) ~= 3 || ~isequal(size(scan3M), size(mask3M))
            error('CERR:InvalidInput', 'CT and mask must be matching 3D arrays.');
        end
        if ~any(mask3M(:))
            error('CERR:EmptyMask', 'Mask is empty for %s.', patientId);
        end

        metadataS = jsondecode(inputS.metadata_json);
        gridS = buildGridStruct(size(scan3M), metadataS);
        featureS = calcRadiomicsForImgType(scan3M, mask3M, paramS, gridS);
        flatS = flattenFeatureStruct(featureS, '');
        if numel(fieldnames(flatS)) ~= expectedFeatures
            error('CERR:FeatureCount', ...
                'Expected %d features, found %d for %s.', ...
                expectedFeatures, numel(fieldnames(flatS)), patientId);
        end
        featureNames = fieldnames(flatS);
        if isempty(referenceFeatureNames)
            referenceFeatureNames = sort(featureNames);
        elseif ~isequal(sort(featureNames), referenceFeatureNames)
            error('CERR:FeatureSchema', ...
                'Feature names differ from the first patient for %s.', patientId);
        end
        flatS.patient_id = patientId;
        maskVoxels = nnz(mask3M);

        outputPath = fullfile(patientDir, [patientId '_radiomics.mat']);
        savePatientResult(outputPath, featureS, flatS, paramS, maskVoxels);
        featureRows{end + 1, 1} = flatS; %#ok<SAGROW>
        logRows{end + 1, 1} = makeLogRow( ...
            patientId, 'success', '', maskVoxels); %#ok<SAGROW>
    catch ME
        message = getReport(ME, 'extended', 'hyperlinks', 'off');
        warning('CERR:PatientFailed', '%s failed: %s', patientId, ME.message);
        logRows{end + 1, 1} = makeLogRow( ...
            patientId, 'error', message, NaN); %#ok<SAGROW>
    end
end

writeStructRowsCsv(featureRows, featureCsv, 'patient_id');
writeStructRowsCsv(logRows, logCsv, 'patient_id');
if numel(featureRows) ~= numel(matFiles)
    error('CERR:IncompleteRun', '%d of %d patients failed.', ...
        numel(matFiles) - numel(featureRows), numel(matFiles));
end
fprintf('Complete: %d patients, %d features.\n', ...
    numel(featureRows), expectedFeatures);


function addCerrToPath(projectRoot)
candidateRoots = {getenv('CERR_ROOT'), ...
    fullfile(projectRoot, 'external', 'CERR'), ...
    fullfile(projectRoot, 'CERR')};
for index = 1:numel(candidateRoots)
    if ~isempty(candidateRoots{index}) && isfolder(candidateRoots{index})
        addpath(genpath(candidateRoots{index}));
        fprintf('Using CERR: %s\n', candidateRoots{index});
        return;
    end
end
error('CERR:NotFound', 'CERR was not found.');
end


function makeDirectory(pathString)
if ~isfolder(pathString)
    mkdir(pathString);
end
end


function gridS = buildGridStruct(scanSizeV, metadataS)
spacingMmV = [1, 1, 1];
if isfield(metadataS, 'radiomics_input_resampling')
    resamplingS = metadataS.radiomics_input_resampling;
    if isfield(resamplingS, 'target_spacing_zyx_mm')
        spacingZyxV = double(resamplingS.target_spacing_zyx_mm(:).');
        if numel(spacingZyxV) == 3
            spacingMmV = [spacingZyxV(3), spacingZyxV(2), spacingZyxV(1)];
        end
    end
end
if any(~isfinite(spacingMmV)) || any(spacingMmV <= 0)
    error('CERR:InvalidSpacing', 'Invalid voxel spacing metadata.');
end

spacingCmV = spacingMmV ./ 10;
gridS = struct();
gridS.PixelSpacingV = spacingCmV;
gridS.xValsV = (0:scanSizeV(1) - 1) .* spacingCmV(1);
gridS.yValsV = (0:scanSizeV(2) - 1) .* spacingCmV(2);
gridS.zValsV = (0:scanSizeV(3) - 1) .* spacingCmV(3);
end


function matFiles = filterMatFiles(matFiles)
patientIds = strtrim(getenv('CERR_PATIENT_IDS'));
patientLimit = str2double(getenv('CERR_PATIENT_LIMIT'));
if ~isempty(patientIds)
    requestedC = strtrim(strsplit(patientIds, ','));
    keepV = false(numel(matFiles), 1);
    for index = 1:numel(matFiles)
        [~, patientId] = fileparts(matFiles(index).name);
        keepV(index) = any(strcmpi(patientId, requestedC));
    end
    matFiles = matFiles(keepV);
end
if ~isnan(patientLimit) && patientLimit > 0 && numel(matFiles) > patientLimit
    matFiles = matFiles(1:patientLimit);
end
end


function flatS = flattenFeatureStruct(inputS, prefix)
flatS = struct();
if ~isstruct(inputS)
    flatS.(cleanFieldName(prefix)) = numericScalar(inputS);
    return;
end

fields = fieldnames(inputS);
for index = 1:numel(fields)
    field = fields{index};
    value = inputS.(field);
    if isempty(prefix)
        childPrefix = field;
    else
        childPrefix = [prefix '_' field]; %#ok<AGROW>
    end
    if isstruct(value)
        childS = flattenFeatureStruct(value, childPrefix);
        childFields = fieldnames(childS);
        for childIndex = 1:numel(childFields)
            flatS.(childFields{childIndex}) = childS.(childFields{childIndex});
        end
    elseif isnumeric(value) || islogical(value)
        flatS.(cleanFieldName(childPrefix)) = numericScalar(value);
    end
end
end


function value = numericScalar(inputValue)
if issparse(inputValue)
    inputValue = full(inputValue);
end
if isscalar(inputValue)
    value = double(inputValue);
else
    value = mean(double(inputValue(:)), 'omitnan');
end
end


function name = cleanFieldName(name)
name = regexprep(name, '[^A-Za-z0-9_]', '_');
if isempty(name)
    name = 'feature';
end
if ~isletter(name(1))
    name = ['f_' name];
end
name = matlab.lang.makeValidName(name);
end


function savePatientResult(pathString, featureS, flatS, paramS, maskVoxels)
temporaryPath = [pathString '.' char(java.util.UUID.randomUUID) '.tmp.mat'];
cleanupObject = onCleanup(@() deleteIfPresent(temporaryPath));
save(temporaryPath, 'featureS', 'flatS', 'paramS', 'maskVoxels');
movefile(temporaryPath, pathString, 'f');
clear cleanupObject;
end


function deleteIfPresent(pathString)
if exist(pathString, 'file') == 2
    delete(pathString);
end
end


function rowS = makeLogRow(patientId, status, message, maskVoxels)
rowS = struct('patient_id', patientId, 'status', status, ...
    'message', message, 'mask_voxels', maskVoxels);
end


function writeStructRowsCsv(rows, csvPath, firstField)
if isempty(rows)
    return;
end
allFields = {};
for index = 1:numel(rows)
    allFields = union(allFields, fieldnames(rows{index}), 'stable');
end
if any(strcmp(allFields, firstField))
    allFields = [{firstField}, ...
        reshape(setdiff(allFields, {firstField}, 'stable'), 1, [])];
end

temporaryPath = [csvPath '.' char(java.util.UUID.randomUUID) '.tmp'];
fid = fopen(temporaryPath, 'w');
if fid < 0
    error('CERR:CsvOpenFailed', 'Could not open CSV: %s', temporaryPath);
end
cleanupObject = onCleanup(@() closeAndDelete(fid, temporaryPath));
fprintf(fid, '%s\n', strjoin(cellfun( ...
    @csvEscape, allFields, 'UniformOutput', false), ','));
for rowIndex = 1:numel(rows)
    values = cell(1, numel(allFields));
    for fieldIndex = 1:numel(allFields)
        field = allFields{fieldIndex};
        if isfield(rows{rowIndex}, field)
            values{fieldIndex} = rows{rowIndex}.(field);
        else
            values{fieldIndex} = NaN;
        end
    end
    fprintf(fid, '%s\n', strjoin(cellfun( ...
        @valueToCsv, values, 'UniformOutput', false), ','));
end
fclose(fid);
fid = -1;
movefile(temporaryPath, csvPath, 'f');
clear cleanupObject;
end


function closeAndDelete(fid, pathString)
if fid >= 0
    try
        fclose(fid);
    catch
    end
end
deleteIfPresent(pathString);
end


function output = valueToCsv(value)
if issparse(value)
    value = full(value);
end
if isnumeric(value) || islogical(value)
    if isempty(value)
        output = '';
    elseif isscalar(value)
        output = sprintf('%.12g', double(value));
    else
        output = csvEscape(mat2str(value));
    end
elseif isstring(value)
    output = csvEscape(char(value));
elseif ischar(value)
    output = csvEscape(value);
else
    output = csvEscape(evalc('disp(value)'));
end
end


function output = csvEscape(value)
if ~ischar(value)
    value = char(value);
end
value = strrep(value, sprintf('\r'), ' ');
value = strrep(value, sprintf('\n'), ' ');
if contains(value, ',') || contains(value, '"')
    value = strrep(value, '"', '""');
    output = ['"' value '"'];
else
    output = value;
end
end
