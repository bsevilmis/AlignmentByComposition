function output = extractObjectProposals(objectProposalParameters, auxiliariesStruct)

% --NOTES
% Comment double for loops inside parfor loops to discard object detection scores. This is needed when the object categories are not one of the 20 PASCAL objects. 
% For Caltech101 dataset we have not generated object detection score maps. We do not have saliency scores either. 
% Uncomment the code sections pertaining to these appropriately.
% Make sure you set the imageExtension correctly.

% --PARAMETER EXTRACTION

% Object proposal extraction parameters
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
k = 200;
minSize = k;
sigma = 0.8;

% HOG extraction parameters
szCell = 8;
nX = 8;
nY = 8;
nDim = nX * nY * 31;
load('bg11.mat');
[bg.R, bg.mu_bg] = whiten(bg, nX, nY);
pixels = double([nY nX] * szCell);
cropsize = ([nY nX]+2) * szCell;

% Box inclusion parameters
lar = objectProposalParameters.areaRatio;
inc = objectProposalParameters.inclusionRatio;
testId = objectProposalParameters.testId;


% --CODE

imageExtension = '.png'; %.jpg, .png

image1 = imread([auxiliariesStruct.mainDatasetFolder '/' auxiliariesStruct.pair1ImageNames{testId} imageExtension]);
image2 = imread([auxiliariesStruct.mainDatasetFolder '/' auxiliariesStruct.pair2ImageNames{testId} imageExtension]);

saliency1 = imread([auxiliariesStruct.saliencyMapsFolder '/' auxiliariesStruct.pair1ImageNames{testId} '.png']);
saliency2 = imread([auxiliariesStruct.saliencyMapsFolder '/' auxiliariesStruct.pair2ImageNames{testId} '.png']);

saliency1 = double(saliency1)/255;
saliency2 = double(saliency2)/255;

% if there is no saliency
%saliency1 = zeros(size(image1,1),size(image1,2));
%saliency2 = zeros(size(image2,1),size(image2,2));

objects1 = load([auxiliariesStruct.objectDetectionsFolder '/' auxiliariesStruct.pair1ImageNames{testId} '.mat']);
objects2 = load([auxiliariesStruct.objectDetectionsFolder '/' auxiliariesStruct.pair2ImageNames{testId} '.mat']);

% if there is no object detection
%objects1 = [];
%objects2 = [];

% Generate selective search object proposals [y1 x1 y2 x2] coordinates
proposals1 = [];
proposals2 = [];

%for j = 1:numel(colorTypes)
%    
%    for jjj = [50 100 200 300]	
%	[boxes, ~, ~, ~] = Image2HierarchicalGrouping(image1, sigma, jjj, jjj, colorTypes{j}, simFunctionHandles);
%    	proposals1 = [proposals1;boxes];
%    
%    	[boxes, ~, ~, ~] = Image2HierarchicalGrouping(image2, sigma, jjj, jjj, colorTypes{j}, simFunctionHandles);
%    	proposals2 = [proposals2;boxes];
%    end
%    
%end

for j = 1:numel(colorTypes)
    
    [boxes, ~, ~, ~] = Image2HierarchicalGrouping(image1, sigma, k, minSize, colorTypes{j}, simFunctionHandles);
    proposals1 = [proposals1;boxes];
    
    [boxes, ~, ~, ~] = Image2HierarchicalGrouping(image2, sigma, k, minSize, colorTypes{j}, simFunctionHandles);
    proposals2 = [proposals2;boxes];
    
end

proposals1 = BoxRemoveDuplicates(proposals1);   %[y1 x1 y2 x2] coordinates
proposals2 = BoxRemoveDuplicates(proposals2);

% Always put the image itself on top
proposals1 = [ [1 1 size(image1,1) size(image1,2)]; proposals1];
proposals2 = [ [1 1 size(image2,1) size(image2,2)]; proposals2];

% Convert to [x1 y1 width height] format
proposals1Converted = [proposals1(:,2) proposals1(:,1) ( proposals1(:,4) - proposals1(:,2) + 1 ) ( proposals1(:,3) - proposals1(:,1) + 1 )];
proposals2Converted = [proposals2(:,2) proposals2(:,1) ( proposals2(:,4) - proposals2(:,2) + 1 ) ( proposals2(:,3) - proposals2(:,1) + 1 )];

% Store area of proposals
areasProposals1 = proposals1Converted(:,3) .* proposals1Converted(:,4);
[~, id_largest1] = max(areasProposals1);

areasProposals2 = proposals2Converted(:,3) .* proposals2Converted(:,4);
[~, id_largest2] = max(areasProposals2);

% Extract unary features from each box
box1Features = cell(size(proposals1,1),1);
box2Features = cell(size(proposals2,1),1);

parfor box1Id = 1:size(proposals1,1)
    
    topLeftRowBox1 = proposals1(box1Id,1);
    topLeftColBox1 = proposals1(box1Id,2);
    bottomRightRowBox1 = proposals1(box1Id,3);
    bottomRightColBox1 = proposals1(box1Id,4);
       
    % saliency score (alternative way)
    numerator = sum(sum(saliency1(proposals1(box1Id,1):proposals1(box1Id,3),proposals1(box1Id,2):proposals1(box1Id,4))));
    denominator = sum(sum(saliency1)) - numerator + (proposals1(box1Id,3) - proposals1(box1Id,1) + 1) * (proposals1(box1Id,4) - proposals1(box1Id,2) + 1);
    saliencyScore = numerator / (denominator+eps);
    
    % detection score
    bestClass = 'background';
    bestClassJaccardIndex = 0;
    
    for class = 1:size(objects1.result,2)
        for detectionId = 1:size(objects1.result(class).detections,1)
            
            detectedObjectMask = objects1.result(class).detections{detectionId};
            if ( size(detectedObjectMask,1) ~= size(image1,1) || size(detectedObjectMask,2) ~= size(image1,2) )
                %                     disp('Resizing object mask...');
                detectedObjectMask = imresize(detectedObjectMask, [size(image1,1) size(image1,2)]);
            end
            
            [rowCoordinates, colCoordinates] = find(detectedObjectMask > 0);
            
            topLeftRowDetection1 = min(rowCoordinates);
            topLeftColDetection1 = min(colCoordinates);
            bottomRightRowDetection1 = max(rowCoordinates);
            bottomRightColDetection1 = max(colCoordinates);
            
            % TODO: We can change this part to make use of rectint
            % function
            if ( ~( (bottomRightRowDetection1 < topLeftRowBox1) || ...
                    (topLeftRowDetection1 > bottomRightRowBox1) || ...
                    (bottomRightColDetection1 < topLeftColBox1) || ...
                    (topLeftColDetection1 > bottomRightColBox1) ) )
                
                proposedBox1 = false(size(image1,1),size(image1,2));
                proposedBox1(proposals1(box1Id,1):proposals1(box1Id,3),proposals1(box1Id,2):proposals1(box1Id,4)) = true;
                numerator = sum(sum( (proposedBox1 & detectedObjectMask )));
                denominator = sum(sum( (proposedBox1 | detectedObjectMask )));
                detectionScore = numerator / denominator;
                                
                if ( detectionScore > bestClassJaccardIndex )
                    bestClassJaccardIndex = detectionScore;
                    bestClass = objects1.result(class).name;
                end
                
            end
            
        end
    end
    
    
    % box inclusion for HOG descriptors
    
    % find larger boxes containing itself
    id_larger = find(lar * areasProposals1 > areasProposals1(box1Id) );
    area_int = rectint(proposals1Converted(box1Id,:), proposals1Converted(id_larger,:));
    IOA = area_int / areasProposals1(box1Id);
    id_valid1 = id_larger(find(IOA >= inc));
    id_valid1 = [id_valid1;id_largest1];
    
    % find smaller boxes contained in itself
    id_smaller = find(areasProposals1 < lar * areasProposals1(box1Id));
    area_int = rectint(proposals1Converted(box1Id,:), proposals1Converted(id_smaller,:));
    IOB = area_int ./ (areasProposals1(id_smaller) )';
    id_valid2 = id_smaller(find(IOB >= inc));
    %         id_smaller = find(areasProposals1 < 0.1 * areasProposals1(box1Id));
    %         area_int = rectint(proposals1Converted(box1Id,:), proposals1Converted(id_smaller,:));
    %         IOB = area_int ./ (areasProposals1(id_smaller) )';
    %         id_valid2 = id_smaller(find(IOB >= inc));
    
    
    % HOG descriptors
    widthPart1 = proposals1Converted(box1Id,3);
    heightPart1 = proposals1Converted(box1Id,4);
    
    padx = szCell * widthPart1 / pixels(2);
    pady = szCell * heightPart1 / pixels(1);
    x1 = round(proposals1(box1Id,2) - padx);
    x2 = round(proposals1(box1Id,4) + padx);
    y1 = round(proposals1(box1Id,1) - pady);
    y2 = round(proposals1(box1Id,3) + pady);
    window1 = subarray(image1, y1, y2, x1, x2, 1);
    patch1 = imresize(window1, cropsize, 'bilinear');
    hog1 = features(double(patch1), szCell);
    hog1 = hog1(:,:,1:end-1);
    hog1 = hog1(:);
    
    A = hog1 - repmat(bg.mu_bg,1,size(hog1,2));
    A = bg.R\(bg.R'\A);
    bias = -A'*bg.mu_bg;
    
    box1Features{box1Id}.coordinates = proposals1(box1Id,:);
    box1Features{box1Id}.saliency = saliencyScore;
    box1Features{box1Id}.detectionClass = bestClass;
    box1Features{box1Id}.detectionClassJaccardIndex = bestClassJaccardIndex;
    
    box1Features{box1Id}.largerBoxIds = id_valid1;
    box1Features{box1Id}.HOG = A;
    box1Features{box1Id}.HOGBias = bias;
        
    box1Features{box1Id}.smallerBoxIds = id_valid2;
    
    if ( mod(box1Id, 100) == 0 )
        disp(['Box1Id: ' num2str(box1Id) ' done...']);
    end
    
end


parfor box2Id = 1:size(proposals2,1)
            
    topLeftRowBox2 = proposals2(box2Id,1);
    topLeftColBox2 = proposals2(box2Id,2);
    bottomRightRowBox2 = proposals2(box2Id,3);
    bottomRightColBox2 = proposals2(box2Id,4);
    
    % alternative way
    numerator = sum(sum(saliency2(proposals2(box2Id,1):proposals2(box2Id,3),proposals2(box2Id,2):proposals2(box2Id,4))));
    denominator = sum(sum(saliency2)) - numerator + (proposals2(box2Id,3) - proposals2(box2Id,1) + 1) * (proposals2(box2Id,4) - proposals2(box2Id,2) + 1);
    saliencyScore = numerator / (denominator+eps);
        
    % detection
    bestClass = 'background';
    bestClassJaccardIndex = 0;
    
    for class = 1:size(objects2.result,2)
        for detectionId = 1:size(objects2.result(class).detections,1)
            
            
            detectedObjectMask = objects2.result(class).detections{detectionId};
            if ( size(detectedObjectMask,1) ~= size(image2,1) || size(detectedObjectMask,2) ~= size(image2,2) )
                %                     disp('Resizing object mask...');
                detectedObjectMask = imresize(detectedObjectMask, [size(image2,1) size(image2,2)]);
            end
            
            [rowCoordinates, colCoordinates] = find(detectedObjectMask > 0);
            
            topLeftRowDetection2 = min(rowCoordinates);
            topLeftColDetection2 = min(colCoordinates);
            bottomRightRowDetection2 = max(rowCoordinates);
            bottomRightColDetection2 = max(colCoordinates);
            
            % TODO: We can change this part to make use of rectint
            % function
            if ( ~( (bottomRightRowDetection2 < topLeftRowBox2) || ...
                    (topLeftRowDetection2 > bottomRightRowBox2) || ...
                    (bottomRightColDetection2 < topLeftColBox2) || ...
                    (topLeftColDetection2 > bottomRightColBox2) ) )
                
                proposedBox2 = false(size(image2,1),size(image2,2));
                proposedBox2(proposals2(box2Id,1):proposals2(box2Id,3),proposals2(box2Id,2):proposals2(box2Id,4)) = true;
                numerator = sum(sum( (proposedBox2 & detectedObjectMask )));
                denominator = sum(sum( (proposedBox2 | detectedObjectMask )));
                detectionScore = numerator / denominator;
                
                if ( detectionScore > bestClassJaccardIndex )
                    bestClassJaccardIndex = detectionScore;
                    bestClass = objects2.result(class).name;
                end
            end
            
        end
    end
    
    
    % box inclusion for HOG descriptors
    
    % find larger boxes containing itself
    id_larger = find(lar * areasProposals2 > areasProposals2(box2Id) );
    area_int = rectint(proposals2Converted(box2Id,:), proposals2Converted(id_larger,:));
    IOA = area_int / areasProposals2(box2Id);
    id_valid1 = id_larger(find(IOA >= inc));
    id_valid1 = [id_valid1;id_largest2];
    
    % find smaller boxes contained in itself
    id_smaller = find(areasProposals2 < lar * areasProposals2(box2Id));
    area_int = rectint(proposals2Converted(box2Id,:), proposals2Converted(id_smaller,:));
    IOB = area_int ./ (areasProposals2(id_smaller) )';
    id_valid2 = id_smaller(find(IOB >= inc));
    %         id_smaller = find(areasProposals2 < 0.1 * areasProposals2(box2Id));
    %         area_int = rectint(proposals2Converted(box2Id,:), proposals2Converted(id_smaller,:));
    %         IOB = area_int ./ (areasProposals2(id_smaller) )';
    %         id_valid2 = id_smaller(find(IOB >= inc));
    
    
    % HOG descriptors
    widthPart2 = proposals2Converted(box2Id,3);
    heightPart2 = proposals2Converted(box2Id,4);
    
    padx = szCell * widthPart2 / pixels(2);
    pady = szCell * heightPart2 / pixels(1);
    x1 = round(proposals2(box2Id,2) - padx);
    x2 = round(proposals2(box2Id,4) + padx);
    y1 = round(proposals2(box2Id,1) - pady);
    y2 = round(proposals2(box2Id,3) + pady);
    window1 = subarray(image2, y1, y2, x1, x2, 1);
    patch1 = imresize(window1, cropsize, 'bilinear');
    hog1 = features(double(patch1), szCell);
    hog1 = hog1(:,:,1:end-1);
    hog1 = hog1(:);
    
    A = hog1 - repmat(bg.mu_bg,1,size(hog1,2));
    A = bg.R\(bg.R'\A);
    bias = -A'*bg.mu_bg;
    
    box2Features{box2Id}.coordinates = proposals2(box2Id,:);
    box2Features{box2Id}.saliency = saliencyScore;
    box2Features{box2Id}.detectionClass = bestClass;
    box2Features{box2Id}.detectionClassJaccardIndex = bestClassJaccardIndex;
    
    box2Features{box2Id}.largerBoxIds = id_valid1;
    box2Features{box2Id}.HOG = A;
    box2Features{box2Id}.HOGBias = bias;
    
    
    box2Features{box2Id}.smallerBoxIds = id_valid2;
    
    
    
    if ( mod(box2Id, 100) == 0 )
        disp(['Box2Id: ' num2str(box2Id) ' done...']);
    end
    
end

output.image1 = image1;
output.image2 = image2;
output.box1Features = box1Features;
output.box2Features = box2Features;


end

% -- HELPFUL FUNCTIONS

function B = subarray(A, i1, i2, j1, j2, pad)

% B = subarray(A, i1, i2, j1, j2, pad)
% Extract subarray from array
% pad with boundary values if pad = 1
% pad with zeros if pad = 0

dim = size(A);
%i1
%i2
is = i1:i2;
js = j1:j2;

if pad,
  is = max(is,1);
  js = max(js,1);
  is = min(is,dim(1));
  js = min(js,dim(2));
  B  = A(is,js,:);
else
  % todo
end

end
