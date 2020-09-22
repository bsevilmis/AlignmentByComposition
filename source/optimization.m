function output = optimization(optimizationParameters, coarseSearchParameters, coarseSearchStruct, objectProposalStruct, colocalizationStruct)

% --PARAMETER EXTRACTION

partBoxIds = coarseSearchStruct.partBoxIds;
partCenterCoordinates = coarseSearchStruct.partCenterCoordinates;
partMaxMatchingScores = coarseSearchStruct.partMaxMatchingScores;
partMatchingScores = coarseSearchStruct.partMatchingScores;
searchGridXCoordinates = coarseSearchStruct.searchGridXCoordinates;
searchGridYCoordinates = coarseSearchStruct.searchGridYCoordinates;
R = coarseSearchStruct.R;
T = coarseSearchStruct.T;

image1 = objectProposalStruct.image1;
image2 = objectProposalStruct.image2;
box1Features = objectProposalStruct.box1Features;

bestId2 = colocalizationStruct.bestId2;
boundingbox2 = objectProposalStruct.box2Features{bestId2}.coordinates;

stride = coarseSearchParameters.stride;
tileStride = stride; % use the same stride size though not necessary

bruteForceSearchSize = optimizationParameters.bruteForceSearchSize;
bruteForceSearchSize = min(bruteForceSearchSize, numel(partBoxIds));

kBestMatches = optimizationParameters.kBestMatches;
kBestMatches = min(size(searchGridXCoordinates,1), kBestMatches);
beamSize = optimizationParameters.beamSize;
sigma = optimizationParameters.sigma;
numberOfBestSolutions = optimizationParameters.numberOfBestSolutions;

fineSearchSigma = optimizationParameters.fineSearchSigma;
fineSearchMaxNumLabels = optimizationParameters.fineSearchMaxNumLabels;

rotationThreshold = optimizationParameters.rotationThreshold;
distortionThreshold = optimizationParameters.distortionThreshold;

% HOG parameters
szCell = 8;
nX=8; nY=8;
nDim = nX*nY*31;

load('bg11.mat');
[bg.R, bg.mu_bg] = whiten(bg,nX,nY);

pixels = double([nY nX] * szCell);
cropsize = ([nY nX]+2) * szCell;


% --CODE

% generate a tiling of the first image to be able to calculate recall of
% the proposals

maxYCoordinateCenter = round(max(partCenterCoordinates(:,1)));
maxYCoordinateCenter = max(1,min(maxYCoordinateCenter, size(image1,1)));
minYCoordinateCenter = round(min(partCenterCoordinates(:,1)));
minYCoordinateCenter = max(1,min(minYCoordinateCenter, size(image1,1)));

maxXCoordinateCenter = round(max(partCenterCoordinates(:,2)));
maxXCoordinateCenter = max(1,min(maxXCoordinateCenter, size(image1,2)));
minXCoordinateCenter = round(min(partCenterCoordinates(:,2)));
minXCoordinateCenter = max(1,min(minXCoordinateCenter, size(image1,2)));

% count how many unique tiles can be filled by proposals from image 1
tileImage = zeros(size(image1,1),size(image1,2));

tileMaskGridX = minXCoordinateCenter : tileStride : maxXCoordinateCenter;
mxGrid = numel(tileMaskGridX) - 1;

tileMaskGridY = minYCoordinateCenter : tileStride : maxYCoordinateCenter;
myGrid = numel(tileMaskGridY) - 1;

gridIndex = 1;
for c = 1:mxGrid
    for r = 1:myGrid
        tileImage(tileMaskGridY(r):tileMaskGridY(r+1), tileMaskGridX(c):tileMaskGridX(c+1)) = gridIndex;
        gridIndex = gridIndex + 1;
    end
end

% check max number of filled tiles if we used all the proposals
uniquelyFilledTileIds = [];

for k = 1:numel(partBoxIds)
    
    currentCenterCoordinates = partCenterCoordinates(k,:);
    currentCenterCoordinatesY = max(1,min(round(currentCenterCoordinates(1)),size(tileImage,1)));
    currentCenterCoordinatesX = max(1,min(round(currentCenterCoordinates(2)),size(tileImage,2)));
    
    uniquelyFilledTileIds = union( uniquelyFilledTileIds, tileImage(currentCenterCoordinatesY, currentCenterCoordinatesX) );
    
end
uniquelyFilledTileNumber = numel(uniquelyFilledTileIds);

% create all possible match candidates to select from
% [order in partBoxId | x | y | matched x | matched y | match score ]
allMatchCandidates = [];
highestMatchScoreIdsForEachPart = [];
for k = 1:numel(partBoxIds)
    
    currentMatchScores = partMatchingScores{k};
    [sortedScores, sortedIndices] = sort(currentMatchScores, 'descend');    
    order = k;
    currentX = partCenterCoordinates(k,2);
    currentY = partCenterCoordinates(k,1);
    
    for j = 1:kBestMatches        
        currentMatchedX = searchGridXCoordinates(sortedIndices(j));
        currentMatchedY = searchGridYCoordinates(sortedIndices(j));
        currentMatchScore = sortedScores(j);
              
        allMatchCandidates = [allMatchCandidates; [order currentX currentY currentMatchedX currentMatchedY currentMatchScore] ];
        
        if ( j == 1 )
            highestMatchScoreIdsForEachPart = [highestMatchScoreIdsForEachPart; size(allMatchCandidates,1)];
        end
        
    end
                   
end

% if allMatchCandidates or highestMatchScoreIdsForEachPart is empty return with a flag
if ( isempty(allMatchCandidates) || isempty(highestMatchScoreIdsForEachPart) )
    output.flag = false;
    return;
end

% initiate beam search from bruteForceSearchSize and from best matching
% parts
[~, sortedTopScoringIndices] = sort( allMatchCandidates(highestMatchScoreIdsForEachPart, 6), 'descend' );
maxMatchingScore = max(cell2mat(partMaxMatchingScores));

beamQueue = [];
resultQueue = [];

for k = 1:bruteForceSearchSize
   
    firstPointX = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) ), 2 );
    firstPointY = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) ), 3 );    
    matchedFirstPointX = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) ), 4 );
    matchedFirstPointY = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) ), 5 );
    firstPointMatchScore = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) ), 6 );
    firstPointMatchScore = firstPointMatchScore / maxMatchingScore;
    
    for l = (k+1):bruteForceSearchSize
        
        secondPointX = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) ), 2 );
        secondPointY = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) ), 3 );        
        matchedSecondPointX = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) ), 4 );
        matchedSecondPointY = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) ), 5 );
        secondPointMatchScore = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) ), 6 );
        secondPointMatchScore = secondPointMatchScore / maxMatchingScore;
        
        for m = (l+1):bruteForceSearchSize
            
            thirdPointX = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) ), 2 );
            thirdPointY = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) ), 3 );            
            matchedThirdPointX = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) ), 4 );
            matchedThirdPointY = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) ), 5 );
            thirdPointMatchScore = allMatchCandidates( highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) ), 6 );
            thirdPointMatchScore = thirdPointMatchScore / maxMatchingScore;
            
            
            AMatrix = [firstPointX firstPointY 1;secondPointX secondPointY 1;thirdPointX thirdPointY 1] \ ...
                [matchedFirstPointX matchedFirstPointY; matchedSecondPointX matchedSecondPointY; matchedThirdPointX, matchedThirdPointY];
                        
%             transposeMatrix = AMatrix';            
%             aValue = transposeMatrix(1,1);
%             dValue = transposeMatrix(2,1);            
%             rotationAngle = abs(atand(dValue/aValue));
           
            
            % ALTERNATIVE ROTATION ANGLE DEFINITION **************************************%
            side1 = [(secondPointX-firstPointX) (secondPointY-firstPointY) 0];
            side2 = [(thirdPointX-secondPointX) (thirdPointY-secondPointY) 0];
            side3 = [(firstPointX-thirdPointX) (firstPointY-thirdPointY) 0];
            
            matchedSide1 = [(matchedSecondPointX-matchedFirstPointX) (matchedSecondPointY-matchedFirstPointY) 0];
            matchedSide2 = [(matchedThirdPointX-matchedSecondPointX) (matchedThirdPointY-matchedSecondPointY) 0];
            matchedSide3 = [(matchedFirstPointX-matchedThirdPointX) (matchedFirstPointY-matchedThirdPointY) 0];
            
            firstRotationAngle = atan2d(norm(cross(side1,matchedSide1)),dot(side1,matchedSide1));
            secondRotationAngle = atan2d(norm(cross(side2,matchedSide2)),dot(side2,matchedSide2));
            thirdRotationAngle = atan2d(norm(cross(side3,matchedSide3)),dot(side3,matchedSide3));
            
            rotationAngle = max(max(firstRotationAngle, secondRotationAngle), thirdRotationAngle);
            %*********************************************************************************%
           
            AMatrix = (AMatrix(1:2,1:2))';
            
            % get distortion score
            try
                [~,S,~] = svd(AMatrix);
                currentDistortionScore = (S(1,1) / S(2,2)) - 1;
                
                if ( rotationAngle > rotationThreshold )
                    currentDistortionScore = Inf;
                end
                                
                if( isinf(currentDistortionScore) || isnan(currentDistortionScore) )
                    currentDistortionScore = Inf;
                end
            catch
                currentDistortionScore = Inf;
            end
            
            
            if ( det(AMatrix) < 0 )
                currentDistortionScore = Inf;
            end
            
            % get convex hull score
            currentXCoordinates = [firstPointX;secondPointX;thirdPointX];
            currentYCoordinates = [firstPointY;secondPointY;thirdPointY];
                                   
            try
                xAllCoordinates = partCenterCoordinates(:,2);
                yAllCoordinates = partCenterCoordinates(:,1);
                
                [~, HullScoreAll] = convhull(xAllCoordinates, yAllCoordinates);
                [~, HullScoreCurrent] = convhull(currentXCoordinates, currentYCoordinates);
                
                currentConvHullScore = HullScoreCurrent / HullScoreAll;
            catch
                currentConvHullScore = 0;
            end
            
            % get average grid score
            roundedXCoordinates = max(1,min(round(currentXCoordinates), size(tileImage,2)));
            roundedYCoordinates = max(1,min(round(currentYCoordinates), size(tileImage,1)));
            
            linearIndices = sub2ind([size(tileImage,1) size(tileImage,2)],roundedYCoordinates, roundedXCoordinates);
            coveredTiles = numel(unique(tileImage(linearIndices)));
            
            currentAverageGridScore = coveredTiles / uniquelyFilledTileNumber;
            
            % get match score
            currentMatchScore = sum([firstPointMatchScore,secondPointMatchScore,thirdPointMatchScore]) / sum(cell2mat(partMaxMatchingScores) / maxMatchingScore);
            
            
            if ( currentDistortionScore > distortionThreshold )
                currentBestTripletScore = -Inf;
            else
                currentBestTripletScore = currentMatchScore + currentConvHullScore + currentAverageGridScore;                
            end
            
            
            % update beamQueue
            if ( ~isinf(currentBestTripletScore) )
%                 disp(['Triplet Score: ' num2str(currentBestTripletScore)]);
                
                beamQueue = updateQueue(beamQueue, currentBestTripletScore,...
                    [highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) );...
                    highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) );...
                    highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) )],...
                    beamSize);
                
                resultQueue = updateQueue(resultQueue, currentBestTripletScore,...
                    [highestMatchScoreIdsForEachPart( sortedTopScoringIndices(k) );...
                    highestMatchScoreIdsForEachPart( sortedTopScoringIndices(l) );...
                    highestMatchScoreIdsForEachPart( sortedTopScoringIndices(m) )],...
                    numberOfBestSolutions);
                                                
            end
            
        end                               
    end                    
end

% if beamQueue or resultQueue is empty return with a flag
if ( isempty(beamQueue) || isempty(resultQueue) )
    output.flag = false;
    return;
end

% do beam search
while(1)
   
    coverageStar = getCoverage(resultQueue, numberOfBestSolutions);
    for j = 1:sigma
       
        beamStarQueue = [];
        while ( ~isempty(beamQueue) )
            
            currentSolution = beamQueue(1,:);
            beamQueue = beamQueue(2:end,:);
            
            distinctlyUsedProposalIds = unique( allMatchCandidates( currentSolution{2}, 1 ) );
            safeSet = find( ~ismember( allMatchCandidates(:,1) , distinctlyUsedProposalIds) );
            scoreVector = -Inf * ones(numel(safeSet),1);
            O = currentSolution{2};
            
            parfor k = 1:numel(safeSet)
               
%                 if ( ~ismember( allMatchCandidates(k,1), distinctlyUsedProposalIds ) )
                   
                    scoreVector(k) = getScoreOfTheSolution( [O;safeSet(k)], allMatchCandidates, tileImage, uniquelyFilledTileNumber, partMaxMatchingScores, maxMatchingScore,...
                        rotationThreshold, distortionThreshold, partCenterCoordinates);
            end
            
            
            [validScores, validScoreIds] = sort(scoreVector, 'descend');
            
            % lets update the beamStarQueue and resultQueue ( assuming
            % beamSize >= numberOfBestSolutions )
            elementsToCheck = min(beamSize, numel(validScores));
            for l = 1:elementsToCheck
                
                score = validScores(l);                
                beamStarQueue = updateQueue(beamStarQueue, score,...
                    [O;safeSet(validScoreIds(l))],...
                    beamSize);
                resultQueue = updateQueue(resultQueue, score,...
                    [O;safeSet(validScoreIds(l))],...
                    numberOfBestSolutions);
                
            end
                                                                                             
%             validScoreIds = find( ~isinf(scoreVector) );
%             
%             for l = 1:numel(validScoreIds)
%             
%                 score = scoreVector(validScoreIds(l));
%                 
%                 beamStarQueue = updateQueue(beamStarQueue, score,...
%                     [O;safeSet(validScoreIds(l))],...
%                     beamSize);
%                 
%                 resultQueue = updateQueue(resultQueue, score,...
%                     [O;safeSet(validScoreIds(l))],...
%                     numberOfBestSolutions);
%                 
%             end
            
            
            
            
%                 end
            
        end
        
        beamQueue = beamStarQueue;
            
    end
    
    disp(['Parts accepted: ' num2str(numel(resultQueue{2}))]);
    
    if ( getCoverage(resultQueue, numberOfBestSolutions) == coverageStar )
        break;
    end
    
end

% Fine grid search using selected object proposals
coarseSolutionProposalIds = zeros(numel(resultQueue{2}),1); 
for k = 1:numel(coarseSolutionProposalIds)    
    coarseSolutionProposalIds(k) = partBoxIds( allMatchCandidates ( resultQueue{2}(k) , 1 ) );        
end
bestMatchLabels = cell( numel(coarseSolutionProposalIds) , 1);

for k = 1:size(bestMatchLabels,1)
   
    part1_1 = box1Features{coarseSolutionProposalIds(k)}.coordinates;
    
    parts1 = repmat({struct('features',[], 'bias',[])},1,1);
    
    
    currentWidth = part1_1(4) - part1_1(2);
    currentHeight = part1_1(3) - part1_1(1);
    
    padx = szCell * currentWidth / pixels(2);
    pady = szCell * currentHeight / pixels(1);
    
    x1 = round(part1_1(2) - padx);
    x2 = round(part1_1(4) + padx);
    y1 = round(part1_1(1) - pady);
    y2 = round(part1_1(3) + pady);
    
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
        
    % estimated part location
    part2_1 = zeros(1,4);
    
    part2_1(2:-1:1) = (R * [part1_1(2);part1_1(1)] + T)';
    part2_1(4:-1:3) = (R * [part1_1(4);part1_1(3)] + T)';
    
    width_part2_1 = part2_1(4) - part2_1(2);
    height_part2_1 = part2_1(3) - part2_1(1);
    
    % sliding window search
    slidingWindowSearchParams = [];
    slidingWindowSearchParams.stride = 1;
    slidingWindowSearchParams.widthPart2 = width_part2_1;
    slidingWindowSearchParams.heightPart2 = height_part2_1;
    slidingWindowSearchParams.boundingBox2 = boundingbox2;
    
    
    % We will constrain the search around the peak response found
    % initially. Let's take fineSearchSigma back and forth        
    slidingWindowSearchParams.boundingBox2 = [allMatchCandidates( resultQueue{2}(k) , 5 ) - stride * fineSearchSigma ,...
        allMatchCandidates( resultQueue{2}(k) , 4 ) - stride * fineSearchSigma,...        
        allMatchCandidates( resultQueue{2}(k) , 5 ) + stride * fineSearchSigma,...
        allMatchCandidates( resultQueue{2}(k) , 4 ) + stride * fineSearchSigma];
    slidingWindowSearchParams.boundingBox2(1) = max(boundingbox2(1),slidingWindowSearchParams.boundingBox2(1));
    slidingWindowSearchParams.boundingBox2(2) = max(boundingbox2(2),slidingWindowSearchParams.boundingBox2(2));
    slidingWindowSearchParams.boundingBox2(3) = min(boundingbox2(3),slidingWindowSearchParams.boundingBox2(3));
    slidingWindowSearchParams.boundingBox2(4) = min(boundingbox2(4),slidingWindowSearchParams.boundingBox2(4));
    
    
    %     tic,
    [heatMap,~] = slidingWindowSearch(parts1,image2,slidingWindowSearchParams);
    %     toc,
    
    % find the local maximums
    localMaximums = findLocalMaximums(heatMap);
    [~, sortedInds] = sort(heatMap(localMaximums),'descend');
    
    coarseSearchPeakResponse = allMatchCandidates( resultQueue{2}(k) , 6 );
    
    % keep maximum of fineSearchMaxNumLabels labels
    labelCounter = 0;
    maxLabelCounter = fineSearchMaxNumLabels;
    
    for l = 1:numel(sortedInds)
        
        if ( heatMap(localMaximums(sortedInds(l))) > coarseSearchPeakResponse && labelCounter < (maxLabelCounter - 1)  )
            
            [myY, myX] = ind2sub([size(heatMap,1), size(heatMap,2)],localMaximums(sortedInds(l)));
            bestMatchLabels{k} = [bestMatchLabels{k}; ...
                [allMatchCandidates( resultQueue{2}(k) , 3 ),allMatchCandidates( resultQueue{2}(k) , 2 ),myY,myX,heatMap(localMaximums(sortedInds(l)))]];
            labelCounter = labelCounter + 1;
        end
        
    end
    
    bestMatchLabels{k} = [bestMatchLabels{k}; ...
        [allMatchCandidates( resultQueue{2}(k) , 3 ),allMatchCandidates( resultQueue{2}(k) , 2 ),allMatchCandidates( resultQueue{2}(k) , 5 ),...
        allMatchCandidates( resultQueue{2}(k) , 4 ),coarseSearchPeakResponse]];
    
    
    if ( size(bestMatchLabels{k},1) == 1 )
        % TODO: for now just add yourself again not to have prob distribution of p = 1;
        bestMatchLabels{k} = [bestMatchLabels{k};bestMatchLabels{k}];
    end
                   
    disp(['Part: ' num2str(k) '/' num2str(size(bestMatchLabels,1)) ' done...']);
                
end


% Run sum-product algorithm to find the best labels
Graph = make_debug_graph(bestMatchLabels);

maxIters = 100;
convTol = 1e-6;

[Graph, iters] = run_loopy_bp_parallel(Graph, maxIters, convTol);
nodeMarginals = get_beliefs(Graph);

% Generate the decision
decisionMatchLabels = zeros(size(bestMatchLabels,1),1);
for k = 1:size(nodeMarginals,1)
    
   [~, idPicked] = max(nodeMarginals{k});
   decisionMatchLabels(k) = idPicked;
    
end


% Return the log score of the solution
logScoreSolution = 0;
for k = 1:size(Graph.fac,1)
    
    ndims = numel(Graph.fac(k).nbrs_var);
    currentDecisions = decisionMatchLabels(Graph.fac(k).nbrs_var);
    
    if ( ndims == 3 )
        logScoreSolution = logScoreSolution + log( Graph.fac(k).p(currentDecisions(1),currentDecisions(2),currentDecisions(3)) );
    else
        logScoreSolution = logScoreSolution + log( Graph.fac(k).p(currentDecisions(1)) );
    end
            
end
disp(['LogScoreSolution: ' num2str(-logScoreSolution)]);


output.bestMatchLabels = bestMatchLabels;
output.decisionMatchLabels = decisionMatchLabels;
output.allMatchCandidates = allMatchCandidates;
output.coarseSolutionProposalIds = coarseSolutionProposalIds;
output.flag = true;


end



function G = make_debug_graph(bestMatchLabels)
    
    G = init_graph();
        
    numberOfNodes = size(bestMatchLabels,1);
    nodeNames = cell(1,numberOfNodes);
    
    for i = 1:numberOfNodes       
        nodeNames{i} = ['x' num2str(i)];        
    end
    
    dims = zeros(1,numberOfNodes);
    
    for i = 1:numberOfNodes
        dims(i) = size(bestMatchLabels{i},1);
    end
        
    ids = zeros(numel(dims),1);
    for i = 1:numel(dims)
        [G,ids(i)] = add_varnode(G,nodeNames{i},dims(i));
    end

    % add potential for nodes
    for i = 1:numberOfNodes
       
        p = bestMatchLabels{i}(:,end);
        p = p ./ sum(p(:));
        G = add_facnode(G, p, ids(i) );
                
    end
    
    % We need to add potential for every triangle
    xCoordinates = [];
    yCoordinates = [];
    
    for i = 1:numberOfNodes
        
        yCoordinates = [yCoordinates; bestMatchLabels{i}(1,1)];
        xCoordinates = [xCoordinates; bestMatchLabels{i}(1,2)];
        
    end
    
    triangles = delaunay(xCoordinates, yCoordinates);
    
    for i = 1:size(triangles,1)
        
        label1Size = numel(bestMatchLabels{triangles(i,1)}(:,end));
        label2Size = numel(bestMatchLabels{triangles(i,2)}(:,end));
        label3Size = numel(bestMatchLabels{triangles(i,3)}(:,end));
        
        p = zeros(label1Size, label2Size, label3Size);
        
        
        firstPointX = xCoordinates(triangles(i,1));
        firstPointY = yCoordinates(triangles(i,1));
        
        secondPointX = xCoordinates(triangles(i,2));
        secondPointY = yCoordinates(triangles(i,2));
        
        thirdPointX = xCoordinates(triangles(i,3));
        thirdPointY = yCoordinates(triangles(i,3));
        
        for label1 = 1:label1Size
            
            matchedFirstPointX = bestMatchLabels{triangles(i,1)}(label1,4);
            matchedFirstPointY = bestMatchLabels{triangles(i,1)}(label1,3);
            
            for label2 = 1:label2Size
                
                matchedSecondPointX = bestMatchLabels{triangles(i,2)}(label2,4);
                matchedSecondPointY = bestMatchLabels{triangles(i,2)}(label2,3);
                
                                
                for label3 = 1:label3Size
                    
                    matchedThirdPointX = bestMatchLabels{triangles(i,3)}(label3,4);
                    matchedThirdPointY = bestMatchLabels{triangles(i,3)}(label3,3);
                    
                    AMatrix = [firstPointX firstPointY 1;secondPointX secondPointY 1;thirdPointX thirdPointY 1] \ ...
                        [matchedFirstPointX matchedFirstPointY; matchedSecondPointX matchedSecondPointY; matchedThirdPointX, matchedThirdPointY];
                    AMatrix = (AMatrix(1:2,1:2))';
                    
                    
                    try
                        [~,S,~] = svd(AMatrix);
                        currentDistortionScore = (S(1,1) / S(2,2)) - 1;
                        if( isinf(currentDistortionScore) || isnan(currentDistortionScore) )
                            currentDistortionScore = Inf;
                        end
                    catch
                        currentDistortionScore = Inf;
                    end
                    
                    
                    if ( det(AMatrix) < 0 )
                        currentDistortionScore = Inf;
                    end
                    
                    p(label1, label2, label3) = currentDistortionScore;
                    
                end
                
            end
            
        end
        
        % normalize p
        sortedDistortions = sort( p(:),'descend' );
        nonInfMaxDistortion = find( ~isinf( sortedDistortions ) );
        nonInfMaxDistortion = sortedDistortions( nonInfMaxDistortion(1) );
        
        p = exp( -(p.^2) ./ (nonInfMaxDistortion * nonInfMaxDistortion) );
        p = p ./ sum(p(:));
        
        G = add_facnode(G,p,ids(triangles(i,1)),ids(triangles(i,2)),ids(triangles(i,3)));
                    
                    
    end
    
        
end

function G = init_graph()
% INIT_GRAPH - Initialize factor graph data structure.
%
% Brown CS242

  G.var = []; % Variable nodes
  G.fac = []; % Factor nodes  
end

function [ var_id ] = get_varid( G, name )
% GET_VARID - Returns the variable index given the name.  This is a naive
% linear-time operation that scans the list of variables.
%
% Brown CS242

  for var_id = 1:numel(G.var)
    if strcmp(lower(G.var(var_id).name), lower(name))
      return;
    end    
  end
  var_id = [];
end

function [G, id] = add_varnode( G, name, dim )
% ADD_VARNODE - Add variable node to factor graph 'G'.
%
% Brown CS242

  id = numel(G.var) + 1;
  
  v.name = name;
%   v.dim = numel(vals);
%   v.vals = vals;
  v.dim = dim;
  v.id = id;
  v.nbrs_fac = [];
  v.observed = 0;
  G.var = cat( 1, G.var, v );
end

function G = add_facnode(G, p, varargin)
% ADD_FACNODE - Add factor node to factor graph 'G'.  
%
% INPUTS:
%   G - Factor graph
%
%   p - Potential matrix with p(i,j,...) the potential for
%       x_a=i, x_b=j, ...
%
%   varargin - Variable nodes involved in factor.  Order matches dimensions
%              potential, e.g. varargin = { a, b, ... } => p(x_a,x_b,...).
%
% Brown CS242

  f.p = p;
  f.nbrs_var = [ varargin{:} ];
  f.id = numel(G.fac) + 1;
  
  % check dimensions
  if ( ( numel(f.nbrs_var) > 1 ) && ( numel(f.nbrs_var) ~= ndims(p)) ) || ...
    ( ( numel(f.nbrs_var) == 1 ) && ( ndims(p) ~= 2 ) )
    error('add_facnode: Factor dimensions does not match size of domain.');
  end
  
  G.fac = cat( 1, G.fac, f );
  for I=f.nbrs_var
    G.var(I).nbrs_fac = [ G.var(I).nbrs_fac; f.id ];
  end
end

function G = add_evidence( G, var_id, val )
% ADD_EVIDENCE - Adds the "evidence" that variable with ID 'var_id' takes
%   on value 'val'.  This slices the factors neighboring var_id accordingly
%   and returns the updated factor graph structure.
%
% Brown CS242

  % iterate factor neighbors
  nbrs = G.var(var_id).nbrs_fac;
  for fac_i = 1:numel(nbrs)
    fac_id = nbrs(fac_i);
    this_fac = G.fac(fac_id);
    
%     % special case: singleton factor
%     if numel(this_fac.nbrs_var) == 1
%       new_p = zeros(numel(this_fac.p), 1);
%       new_p(val) = 1;
%       this_fac.p = new_p;
%       continue;
%     end
    
    % slice factor
    I = find( this_fac.nbrs_var == var_id );
    index = repmat( {':'}, numel(this_fac.nbrs_var), 1 );
    index(I) = {val};
    new_p = this_fac.p( index{:} );
    new_nbrs_var = this_fac.nbrs_var;
    new_nbrs_var(I) = [];
    
    % adjust dimensions
    if I==1
      new_p = shiftdim(new_p, 1);
    else
      new_p = squeeze( new_p );
    end
    
    % save factor
    this_fac.p = squeeze( new_p );    
    this_fac.nbrs_var = new_nbrs_var;
    G.fac(fac_id) = this_fac;
  end
  
  % remove edges from var node
  G.var(var_id).nbrs_fac = [];  
  G.var(var_id).observed = val;
end

function [node_marg] = get_beliefs(G)
% GET_BELIEFS - Returns cell arrays containing beliefs for each node.
%
% Inputs:
%   G: Factor graph to perform loopy BP over
%
% Outputs:
%   node_marg: cell array containing the marginals of each variable node,
%              computed by multiplying and normalizing the current messages.
%              Same format as nodeMarg = marg_brute_force(G);
%
% Brown CS242

  num_var = numel(G.var);
  node_marg = cell(num_var,1);
  % FILL ME IN!
  for i = 1:numel(G.var)
      variable_id = G.var(i).id;
      factor_indices = G.var(i).nbrs_fac;
      node_marg{i} = ones(G.var(i).dim,1);
      
      % multiply factor messages
      for j = 1:numel(factor_indices)
          message_index = find(G.fac(factor_indices(j)).nbrs_var == variable_id);
          message = G.fac(factor_indices(j)).message{message_index};
          norm_const = G.fac(factor_indices(j)).norm_const{message_index};
          message = message*norm_const;
          
          assert(abs(sum(message)- 1) < 1e-6); % check if messages are properly normalized
          
          node_marg{i} = node_marg{i}.*message;
      end
      
      % normalize
      node_marg{i} = node_marg{i} ./ sum(node_marg{i});
  
  end

end

function [G, iters] = run_loopy_bp_parallel( G, max_iters, conv_tol )
% RUN_LOOPY_BP - Runs Loopy Belief Propagation (Sum-Product) on a factor 
%   Graph given by 'G'.This implements a "parallel" updating scheme in
%   which all factor-to-variable messages are updated in a single clock
%   cycle, and then all variable-to-factor messages are updated.
%
% Inputs:
%   G: Factor graph to perform loopy BP over
%   max_iters: Maximum number of iterations to run loopy BP for
%   conv_tol:  Convergence tolerance (threshold on max message change)
%
% Outputs:
%   G: The factor graph, with messages. Note that G will be passed to
%      get_beliefs.m to compute the node marginals.
%
%   iters: How many iterations it took for loopy BP to converge
%
% Brown CS242
  

% Initialize messages, by adding a message and normalization constant
% fields for factors and variables
G = initialize_messages(G);

% Main Loop
for iters = 1:max_iters
    % FILL ME IN!
    
    % save a copy of the graph
    Gprev = G;
    
    % one pass of message-passing from factors to variables and from
    % variables to factors
    G = pass_message_factor_to_variable(G);
    G = pass_message_variable_to_factor(G);
    
    % check if message-passing converged by comparing messages of Gprev and
    % G
    convergence_flag = check_convergence(Gprev,G,conv_tol);
    
    % if message-passing converged, exit the loop
    if(convergence_flag)
        break;
    end
    
end
  
end

function G = initialize_messages(G)

% add message field & normalization constants
[G.var(:).message] = deal({});
[G.fac(:).message] = deal({});
[G.var(:).norm_const] = deal({});
[G.fac(:).norm_const] = deal({});

% initialize variable to factor messages
for i = 1:size(G.var,1)
    for j = 1:numel(G.var(i).nbrs_fac)
        message = ones(G.var(i).dim,1);
        norm_const = 1/sum(message);
        
        G.var(i).message = cat( 2, G.var(i).message, message );
        G.var(i).norm_const = cat( 2, G.var(i).norm_const, norm_const );
        
    end
end

% initialize factor to variable messages
for i = 1:size(G.fac,1)
    for j = 1:numel(G.fac(i).nbrs_var)
        %message = zeros(size(G.fac(i).p,j),1);
        message = ones(size(G.fac(i).p,j),1);
        norm_const = 1/sum(message);
                
        G.fac(i).message = cat( 2, G.fac(i).message, message );
        G.fac(i).norm_const = cat( 2, G.fac(i).norm_const, norm_const );
        
    end
end

end

function G = pass_message_factor_to_variable(G)

for i = 1:size(G.fac,1)
    factor_id = G.fac(i).id;
    variable_ids = G.fac(i).nbrs_var;
    
    for j = 1:numel(G.fac(i).nbrs_var)
        current_variable_id = G.fac(i).nbrs_var(j);
        neighbor_variable_ids = setdiff(variable_ids,current_variable_id);
        potential = G.fac(i).p;
        
        % multiply
        for k = 1:numel(neighbor_variable_ids)
            factor_index = find(G.var(neighbor_variable_ids(k)).nbrs_fac == factor_id);
            current_message = G.var(neighbor_variable_ids(k)).message{factor_index};
            current_norm_const = G.var(neighbor_variable_ids(k)).norm_const{factor_index};
            current_message = current_norm_const * current_message;
            
            shape_order = ones(1,numel(size(potential)));
            shape_order(find(variable_ids ==  neighbor_variable_ids(k))) = numel(current_message);
            
            potential = bsxfun(@times,potential,reshape(current_message,shape_order));
        end
        
        % sum
        for k = 1:numel(neighbor_variable_ids)
            potential = sum(potential,find(variable_ids ==  neighbor_variable_ids(k)));
        end
        
        potential = reshape(potential,[G.var(current_variable_id).dim,1]);
        
        % set message & normalization constant
        G.fac(factor_id).message{j} = potential;
        G.fac(factor_id).norm_const{j} = 1/sum(potential);
    end
end
        
      
end

function G = pass_message_variable_to_factor(G)

for i = 1:size(G.var,1)
   variable_id = G.var(i).id;
   factor_ids = G.var(i).nbrs_fac;
   
   for j = 1:numel(G.var(i).nbrs_fac)
       current_factor_id = G.var(i).nbrs_fac(j);
       neighbor_factor_ids = setdiff(factor_ids,current_factor_id);
       potential = ones(G.var(i).dim,1);
       
       % multiply
       for k = 1:numel(neighbor_factor_ids)
           variable_index = find(G.fac(neighbor_factor_ids(k)).nbrs_var == variable_id);
           current_message = G.fac(neighbor_factor_ids(k)).message{variable_index};
           current_norm_const = G.fac(neighbor_factor_ids(k)).norm_const{variable_index};
           current_message = current_norm_const * current_message;
           
           potential = bsxfun(@times,potential,current_message);
       end
       
       
       % set message & normalization constant
       G.var(variable_id).message{j} = potential;
       G.var(variable_id).norm_const{j} = 1/sum(potential);
       
   end   
end

end

function convergence_flag = check_convergence(Gprev,G,conv_tol)

max_changes = [];

% change in variable to factor messages
for i = 1:size(G.var,1)
    for j = 1:size(G.var(i).message,2)
       old_message = Gprev.var(i).message{j}*Gprev.var(i).norm_const{j};
       current_message = G.var(i).message{j}*G.var(i).norm_const{j};
       max_changes = [max_changes;max(abs(old_message-current_message))];
    end
end

% change in factor to variable messages
for i = 1:size(G.fac,1)
    for j = 1:size(G.fac(i).message,2)
       old_message = Gprev.fac(i).message{j}*Gprev.fac(i).norm_const{j};
       current_message = G.fac(i).message{j}*G.fac(i).norm_const{j};
       max_changes = [max_changes;max(abs(old_message-current_message))];
    end
end

% if the maximum change is below convergence tolerance return true
if(max(max_changes) <= conv_tol)
    convergence_flag = true;
else
    convergence_flag = false;
end


end


function localMaximums = findLocalMaximums(heatMap)

% localMaximums are the linear indices of the maximum points

% find the seeds
[yPoints, xPoints] = find(heatMap > 0);


% assume every seed returns a maximum point
localMaximums = zeros(numel(yPoints),1);
% percentageCounter = 10:10:100;


for i = 1:numel(yPoints)
    
    currentPoint = [yPoints(i), xPoints(i)];
    peakLinearIndex = doGradientAscent(heatMap, currentPoint);
    localMaximums(i) = peakLinearIndex;
    
%     if ( ( i/numel(yPoints) ) * 100 >= percentageCounter(1) )
%         disp([num2str(percentageCounter(1)) '% peaks done...']);
%         percentageCounter(1) = [];
%     end
    
end


localMaximums = unique(localMaximums);

end

function peakLinearIndex = doGradientAscent(heatMap, currentPoint)

converged = false;
bestPoint = currentPoint;

while (~converged)

    currentPoint = bestPoint;
    currentPeakValue = heatMap(currentPoint(1), currentPoint(2)); % for double check
    
    for rowChange = -1:1:1
        for colChange = -1:1:1
            
            nextPoint = [currentPoint(1) + rowChange, currentPoint(2) + colChange];
            nextPoint(1) = max(1,min(size(heatMap,1),nextPoint(1)));
            nextPoint(2) = max(1,min(size(heatMap,2),nextPoint(2)));
            
            if ( heatMap( nextPoint(1), nextPoint(2) ) > currentPeakValue )
                
                currentPeakValue = heatMap( nextPoint(1), nextPoint(2) );
                bestPoint = nextPoint;
                
            end
            
        end
    end
    
    if ( isequal(currentPoint, bestPoint) )
        converged = true;
    end
    
    
end


peakLinearIndex = sub2ind([size(heatMap,1), size(heatMap,2)], bestPoint(1), bestPoint(2));


end


function score = getScoreOfTheSolution( O, allMatchCandidates, tileImage, uniquelyFilledTileNumber, partMaxMatchingScores, maxMatchingScore,...
    rotationThreshold, distortionThreshold, partCenterCoordinates)


matchScores = allMatchCandidates(O, 6) / maxMatchingScore;
matchScore = sum(matchScores) / sum(cell2mat(partMaxMatchingScores) / maxMatchingScore);

xCoordinates = allMatchCandidates( O, 2 );
yCoordinates = allMatchCandidates( O, 3 );
triangles = delaunay(xCoordinates, yCoordinates);

if ( numel(unique(sort(triangles(:)))) ~= numel(O) )
    % we have duplicate points do not consider them
    score = -Inf;
    return;
end

distortionScores = zeros(1,size(triangles,1));
rotationAngles = zeros(1,size(triangles,1));

for i = 1:size(triangles,1)
    
    firstPointX = xCoordinates(triangles(i,1));
    firstPointY = yCoordinates(triangles(i,1));
    
    matchedFirstPointX = allMatchCandidates( O(triangles(i,1)), 4);
    matchedFirstPointY = allMatchCandidates( O(triangles(i,1)), 5);
    
    secondPointX = xCoordinates(triangles(i,2));
    secondPointY = yCoordinates(triangles(i,2));

    matchedSecondPointX = allMatchCandidates( O(triangles(i,2)), 4);
    matchedSecondPointY = allMatchCandidates( O(triangles(i,2)), 5);
    
    thirdPointX = xCoordinates(triangles(i,3));
    thirdPointY = yCoordinates(triangles(i,3));

    matchedThirdPointX = allMatchCandidates( O(triangles(i,3)), 4);
    matchedThirdPointY = allMatchCandidates( O(triangles(i,3)), 5);
    
    AMatrix = [firstPointX firstPointY 1;secondPointX secondPointY 1;thirdPointX thirdPointY 1] \ ...
        [matchedFirstPointX matchedFirstPointY; matchedSecondPointX matchedSecondPointY; matchedThirdPointX, matchedThirdPointY];
    
%     transposeMatrix = AMatrix';
% 
%     aValue = transposeMatrix(1,1);
%     dValue = transposeMatrix(2,1);
%     
%     rotationAngle = abs(atand(dValue/aValue));
    
    
    % ALTERNATIVE ROTATION ANGLE DEFINITION **************************************%
    side1 = [(secondPointX-firstPointX) (secondPointY-firstPointY) 0];
    side2 = [(thirdPointX-secondPointX) (thirdPointY-secondPointY) 0];
    side3 = [(firstPointX-thirdPointX) (firstPointY-thirdPointY) 0];
    
    matchedSide1 = [(matchedSecondPointX-matchedFirstPointX) (matchedSecondPointY-matchedFirstPointY) 0];
    matchedSide2 = [(matchedThirdPointX-matchedSecondPointX) (matchedThirdPointY-matchedSecondPointY) 0];
    matchedSide3 = [(matchedFirstPointX-matchedThirdPointX) (matchedFirstPointY-matchedThirdPointY) 0];
    
    firstRotationAngle = atan2d(norm(cross(side1,matchedSide1)),dot(side1,matchedSide1));
    secondRotationAngle = atan2d(norm(cross(side2,matchedSide2)),dot(side2,matchedSide2));
    thirdRotationAngle = atan2d(norm(cross(side3,matchedSide3)),dot(side3,matchedSide3));
    
    rotationAngle = max(max(firstRotationAngle, secondRotationAngle), thirdRotationAngle);
    %*********************************************************************************%

    rotationAngles(i) = rotationAngle;
        
    AMatrix = (AMatrix(1:2,1:2))';
    
    
    try
        [~,S,~] = svd(AMatrix);
        currentDistortionScore = (S(1,1) / S(2,2)) - 1;
        
        
        % check rotation (if rotation > 10 degrees then discard)
        if ( rotationAngle > rotationThreshold )
            currentDistortionScore = Inf;
        end
                        
        if( isinf(currentDistortionScore) || isnan(currentDistortionScore) )
            currentDistortionScore = Inf;
        end
    catch
        currentDistortionScore = Inf;
    end
    
    
    if ( det(AMatrix) < 0 )
        currentDistortionScore = Inf;
    end
    
    
    distortionScores(i) = currentDistortionScore;
    
end

maxDistortionScore = max(distortionScores);

% get grid score
roundedXCoordinates = max(1,min(round(xCoordinates), size(tileImage,2)));
roundedYCoordinates = max(1,min(round(yCoordinates), size(tileImage,1)));

linearIndices = sub2ind([size(tileImage,1) size(tileImage,2)],roundedYCoordinates, roundedXCoordinates);
coveredTiles = numel(unique(tileImage(linearIndices)));

averageGridScore = coveredTiles / uniquelyFilledTileNumber;

% get convex hull score
try
xAllCoordinates = partCenterCoordinates(:,2);
yAllCoordinates = partCenterCoordinates(:,1);   
    
[~, HullScoreAll] = convhull(xAllCoordinates, yAllCoordinates);        
[~, HullScoreCurrent] = convhull(xCoordinates, yCoordinates);

convHullScore = HullScoreCurrent / HullScoreAll;
catch
    convHullScore = 0;
end


if ( maxDistortionScore > distortionThreshold )
    score = -Inf;
else
    score = matchScore + convHullScore + averageGridScore;
end
    
end


function coverage = getCoverage(queue, numberOfNeighbors)

if( size(queue,1) < numberOfNeighbors )
    coverage = Inf;
else
    coverage = queue{1,1};
end

end


function queue = updateQueue(queue, score, pointSet, allowedSize)


if ( size(queue,1) == allowedSize )
    
    if ( score > queue{1,1} )
        
        if ( size(queue,1) >= 2 )
        
            queue = queue(2:end,:);
            queue = [ {score, pointSet}; queue];
            
            [~, ids] = sort( cell2mat(queue(:,1)), 'ascend');
            queue = queue(ids,:);
        
        else
            
            queue = {score, pointSet};
            
        end
        
    end
    
else
    
    queue = [ {score, pointSet}; queue];
        
    [~, ids] = sort( cell2mat(queue(:,1)), 'ascend');
    queue = queue(ids,:);
    
end

end

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
