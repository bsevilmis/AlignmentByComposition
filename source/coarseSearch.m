function output = coarseSearch(coarseSearchParameters, objectProposalStruct, colocalizationStruct)

% --PARAMETER EXTRACTION

image1 = objectProposalStruct.image1;
image2 = objectProposalStruct.image2;

bestId1 = colocalizationStruct.bestId1;
bestId2 = colocalizationStruct.bestId2;

boundingbox1 = objectProposalStruct.box1Features{bestId1}.coordinates;
boundingbox2 = objectProposalStruct.box2Features{bestId2}.coordinates;

box1Features = objectProposalStruct.box1Features;

stride = coarseSearchParameters.stride;

% HOG parameters
szCell = 8;
nX=8; nY=8;
nDim = nX*nY*31;

load('bg11.mat');
[bg.R, bg.mu_bg] = whiten(bg,nX,nY);

pixels = double([nY nX] * szCell);
cropsize = ([nY nX]+2) * szCell;


% --CODE

tx = boundingbox2(2) - ( boundingbox1(2)*(boundingbox2(4)-boundingbox2(2)) / ( (boundingbox1(4)-boundingbox1(2)) ) );
ty = boundingbox2(1) - ( boundingbox1(1)*(boundingbox2(3)-boundingbox2(1)) / ( (boundingbox1(3)-boundingbox1(1)) ) );
sx = ( (boundingbox2(4)-boundingbox2(2)) / ( (boundingbox1(4)-boundingbox1(2)) ) );
sy = ( (boundingbox2(3)-boundingbox2(1)) / ( (boundingbox1(3)-boundingbox1(1)) ) );

R = [sx 0;0 sy];
T = [tx;ty];

[searchGridX, searchGridY] = meshgrid(1:stride:size(image2,2), 1:stride:size(image2,1));
linearSearchIndices = find(searchGridX >= boundingbox2(2) & searchGridX <= boundingbox2(4) & ...
    searchGridY >= boundingbox2(1) & searchGridY <= boundingbox2(3) );
searchGridXCoordinates = searchGridX(linearSearchIndices);
searchGridYCoordinates = searchGridY(linearSearchIndices);

partBoxIds = box1Features{bestId1}.smallerBoxIds;

partMatchingScores = cell(numel(partBoxIds), 1);
partMaxMatchingScores = cell(numel(partBoxIds), 1);
partCenterCoordinates = zeros(numel(partBoxIds), 2);

for k = 1:numel(partBoxIds)
        
    part1 = box1Features{partBoxIds(k)}.coordinates; %[y1 x1 y2 x2]
    
    parts1 = repmat({struct('features',[], 'bias',[])},1,1);
        
    currentWidth = part1(4) - part1(2);
    currentHeight = part1(3) - part1(1);
    
    padx = szCell * currentWidth / pixels(2);
    pady = szCell * currentHeight / pixels(1);
    
    x1 = round(part1(2) - padx);
    x2 = round(part1(4) + padx);
    y1 = round(part1(1) - pady);
    y2 = round(part1(3) + pady);
    
    window = subarray(image1, y1, y2, x1, x2, 1);
    patch = imresize(window, cropsize, 'bilinear');
    hog = features(double(patch), szCell);
    hog = hog(:,:,1:end-1);
    hog = hog(:);
    
    A = hog - repmat(bg.mu_bg,1,size(hog,2));
    A = bg.R\(bg.R'\A);
    bias = -A'*bg.mu_bg;
    
    parts1{1}.features = A;
    parts1{1}.bias = bias;
    
    %            figure(200), imshow(image1), hold on,...
    %                rectangle('Position',[part1_1(2),part1_1(1),...
    %                part1_1(4)-part1_1(2)+1,part1_1(3)-part1_1(1)+1],'EdgeColor','r','LineWidth',1);
    
    % estimated part location
    part2 = zeros(1,4);
    
    part2(2:-1:1) = (R * [part1(2);part1(1)] + T)';
    part2(4:-1:3) = (R * [part1(4);part1(3)] + T)';
    
    width_part2_1 = part2(4) - part2(2);
    height_part2_1 = part2(3) - part2(1);
    
    % sliding window search
    slidingWindowSearchParams = [];
    slidingWindowSearchParams.stride = stride;
    slidingWindowSearchParams.widthPart2 = width_part2_1;
    slidingWindowSearchParams.heightPart2 = height_part2_1;
    slidingWindowSearchParams.boundingBox2 = boundingbox2;
    
    tic,
    [heatMap,~] = slidingWindowSearch(parts1,image2,slidingWindowSearchParams);
    toc,
    
    disp(['Part: ' num2str(k) '/' num2str(numel(partBoxIds)) ' done...']);
    
    
    partMatchingScores{k} = heatMap(sub2ind([size(image2,1) size(image2,2)], searchGridYCoordinates, searchGridXCoordinates));
    partMaxMatchingScores{k} = max(partMatchingScores{k});
    
    partCenterCoordinates(k,:) = [0.5 * (part1(1) + part1(3)), ...
        0.5 * (part1(2) + part1(4))];
    
end

output.partBoxIds = partBoxIds;
output.partMatchingScores = partMatchingScores;
output.partMaxMatchingScores = partMaxMatchingScores;
output.partCenterCoordinates = partCenterCoordinates;
output.searchGridXCoordinates = searchGridXCoordinates;
output.searchGridYCoordinates = searchGridYCoordinates;
output.R = R;
output.T = T;



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