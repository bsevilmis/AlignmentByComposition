function [] = CaltechEvaluation()

% Measure Caltech101 performance

clc; clear all; close all;

p2m = @(X,Y,M,N) poly2mask(X,Y,M,N); % use MATLAB's fn

dataset = 'Caltech101';

resultsDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/results/' dataset];
% resultsDirectory = ['/home/berksevilmis/.gvfs/SFTP for bsevilmi on ssh.ccv.brown.edu/gpfs_home/bsevilmi/scratch/AlignmentByComposition/results/' dataset];

groundTruthFileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/gt.txt'];
% groundTruthFileName = ['/home/berksevilmis/.gvfs/SFTP for bsevilmi on ssh.ccv.brown.edu/gpfs_home/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/gt.txt'];
groundTruthFid = fopen(groundTruthFileName, 'r');
groundTruthCell = textscan(groundTruthFid, '%s %s\n');
fclose(groundTruthFid);

% groundTruthKeypointsDirectory = ['/home/berksevilmis/.gvfs/SFTP for bsevilmi on ssh.ccv.brown.edu/gpfs_home/bsevilmi/scratch/AlignmentByComposition/datasets/GroundTruth/' dataset];
groundTruthKeypointsDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/GroundTruth/' dataset];

CaltechScoresLinearACC = [];
CaltechScoresLinearBilateralFilteredACC = [];
CaltechScoresDeformableACC = [];
CaltechScoresDeformableBilateralFilteredACC = [];
CaltechScoresTPSACC = [];
CaltechScoresNaturalACC = [];

CaltechScoresLinearIOU = [];
CaltechScoresLinearBilateralFilteredIOU = [];
CaltechScoresDeformableIOU = [];
CaltechScoresDeformableBilateralFilteredIOU = [];
CaltechScoresTPSIOU = [];
CaltechScoresNaturalIOU = [];

CaltechScoresLinearLOCERR = [];
CaltechScoresLinearBilateralFilteredLOCERR = [];
CaltechScoresDeformableLOCERR = [];
CaltechScoresDeformableBilateralFilteredLOCERR = [];
CaltechScoresTPSLOCERR = [];
CaltechScoresNaturalLOCERR = [];

for i = 1:1515
    
    if ( strcmp(groundTruthCell{1}{i}, 'airplanes_annotation_0118') || ...
            strcmp(groundTruthCell{1}{i}, 'airplanes_annotation_0202') || ...
            strcmp(groundTruthCell{2}{i}, 'airplanes_annotation_0118') || ...
            strcmp(groundTruthCell{2}{i}, 'airplanes_annotation_0202') )
        % these annotations have problems
        continue;
    end
    
    currentResult = [];
    myAnno1 = [];
    myAnno2 = [];
    acc = [];
    iou = [];
    loc_err = [];
    annA = [];
    annB = [];
    anno1 = [];
    anno2 = [];
    
    
    currentResult = load( [resultsDirectory '/' num2str(i) '.mat'] );
    
    myAnno1 = imread( [groundTruthKeypointsDirectory '/' groundTruthCell{1}{i} '.png'] );
    myAnno2 = imread( [groundTruthKeypointsDirectory '/' groundTruthCell{2}{i} '.png'] );
    
    
    underscoreIndex1 = strfind(groundTruthCell{1}{i}, '_');
    class1 = groundTruthCell{1}{i}(1:underscoreIndex1(end-1)-1);
    annotationName1 = groundTruthCell{1}{i}(underscoreIndex1(end-1)+1:end);
    

       
    underscoreIndex2 = strfind(groundTruthCell{2}{i}, '_');
    class2 = groundTruthCell{2}{i}(1:underscoreIndex2(end-1)-1);
    annotationName2 = groundTruthCell{2}{i}(underscoreIndex2(end-1)+1:end);
    

         
    annA = load([groundTruthKeypointsDirectory '/Annotations/' class1 '/' annotationName1 '.mat']);
    annB = load([groundTruthKeypointsDirectory '/Annotations/' class2 '/' annotationName2 '.mat']);
    
    % Extract mask images from polygonal annotations
    anno1 = p2m(annA.obj_contour(1,:)+annA.box_coord(3), annA.obj_contour(2,:)+annA.box_coord(1), size(myAnno1,1), size(myAnno1,2));
    anno2 = p2m(annB.obj_contour(1,:)+annB.box_coord(3), annB.obj_contour(2,:)+annB.box_coord(1), size(myAnno2,1), size(myAnno2,2));
    
    % check if annotation area is insignificant (<1%)
    if length(find(anno1(:)~=0))/length(anno1(:))<0.01 || length(find(anno2(:)~=0))/length(anno2(:))<0.01
        continue;
    end
    
    [acc, iou, loc_err] = getPerformance(anno1, anno2, round(currentResult.flowEstimationStruct.flowLinearX), round(currentResult.flowEstimationStruct.flowLinearY));
    
    if ( ~(isnan(acc) || isnan(iou) || isnan(loc_err)) )
        
        
        CaltechScoresLinearACC = [CaltechScoresLinearACC; acc];
        CaltechScoresLinearIOU = [CaltechScoresLinearIOU; iou];
        CaltechScoresLinearLOCERR = [CaltechScoresLinearLOCERR; loc_err];
        
    end
    
    [acc, iou, loc_err] = getPerformance(anno1, anno2, round(currentResult.flowEstimationStruct.flowLinearXBilateralFiltered), round(currentResult.flowEstimationStruct.flowLinearYBilateralFiltered));
    
    if ( ~(isnan(acc) || isnan(iou) || isnan(loc_err)) )
        
        CaltechScoresLinearBilateralFilteredACC = [CaltechScoresLinearBilateralFilteredACC; acc];
        CaltechScoresLinearBilateralFilteredIOU = [CaltechScoresLinearBilateralFilteredIOU; iou];
        CaltechScoresLinearBilateralFilteredLOCERR = [CaltechScoresLinearBilateralFilteredLOCERR; loc_err];
        
    end
    
    [acc, iou, loc_err] = getPerformance(anno1, anno2, round(currentResult.flowEstimationStruct.flowDeformableX), round(currentResult.flowEstimationStruct.flowDeformableY));
    
    if ( ~(isnan(acc) || isnan(iou) || isnan(loc_err)) )
        
        CaltechScoresDeformableACC = [CaltechScoresDeformableACC; acc];
        CaltechScoresDeformableIOU = [CaltechScoresDeformableIOU; iou];
        CaltechScoresDeformableLOCERR = [CaltechScoresDeformableLOCERR; loc_err];
        
    end
    
    [acc, iou, loc_err] = getPerformance(anno1, anno2, round(currentResult.flowEstimationStruct.flowDeformableXBilateralFiltered), round(currentResult.flowEstimationStruct.flowDeformableYBilateralFiltered));
    
    if ( ~(isnan(acc) || isnan(iou) || isnan(loc_err)) )
        
        CaltechScoresDeformableBilateralFilteredACC = [CaltechScoresDeformableBilateralFilteredACC; acc];
        CaltechScoresDeformableBilateralFilteredIOU = [CaltechScoresDeformableBilateralFilteredIOU; iou];
        CaltechScoresDeformableBilateralFilteredLOCERR = [CaltechScoresDeformableBilateralFilteredLOCERR; loc_err];
        
    end
    
    [acc, iou, loc_err] = getPerformance(anno1, anno2, round(currentResult.flowEstimationStruct.flowTPSX), round(currentResult.flowEstimationStruct.flowTPSY));
    
    if ( ~(isnan(acc) || isnan(iou) || isnan(loc_err)) )
        
        CaltechScoresTPSACC = [CaltechScoresTPSACC; acc];
        CaltechScoresTPSIOU = [CaltechScoresTPSIOU; iou];
        CaltechScoresTPSLOCERR = [CaltechScoresTPSLOCERR; loc_err];
        
    end
    
    [acc, iou, loc_err] = getPerformance(anno1, anno2, round(currentResult.flowEstimationStruct.flowNaturalX), round(currentResult.flowEstimationStruct.flowNaturalY));
    
    if ( ~(isnan(acc) || isnan(iou) || isnan(loc_err)) )
        
        CaltechScoresNaturalACC = [CaltechScoresNaturalACC; acc];
        CaltechScoresNaturalIOU = [CaltechScoresNaturalIOU; iou];
        CaltechScoresNaturalLOCERR = [CaltechScoresNaturalLOCERR; loc_err];
        
    end
    
    disp([num2str(i) ' done..']);
    
    
end

disp([dataset ' mean ACC(Linear): ' num2str(mean(CaltechScoresLinearACC))]);
disp([dataset ' mean ACC(LinearBilateralFiltered): ' num2str(mean(CaltechScoresLinearBilateralFilteredACC))]);
disp([dataset ' mean ACC(Deformable): ' num2str(mean(CaltechScoresDeformableACC))]);
disp([dataset ' mean ACC(DeformableBilateralFiltered): ' num2str(mean(CaltechScoresDeformableBilateralFilteredACC))]);
disp([dataset ' mean ACC(TPS): ' num2str(mean(CaltechScoresTPSACC))]);
disp([dataset ' mean ACC(Natural): ' num2str(mean(CaltechScoresNaturalACC))]);

disp(' ');

disp([dataset ' mean IOU(Linear): ' num2str(mean(CaltechScoresLinearIOU))]);
disp([dataset ' mean IOU(LinearBilateralFiltered): ' num2str(mean(CaltechScoresLinearBilateralFilteredIOU))]);
disp([dataset ' mean IOU(Deformable): ' num2str(mean(CaltechScoresDeformableIOU))]);
disp([dataset ' mean IOU(DeformableBilateralFiltered): ' num2str(mean(CaltechScoresDeformableBilateralFilteredIOU))]);
disp([dataset ' mean IOU(TPS): ' num2str(mean(CaltechScoresTPSIOU))]);
disp([dataset ' mean IOU(Natural): ' num2str(mean(CaltechScoresNaturalIOU))]);

disp(' ');

disp([dataset ' mean LOC_ERR(Linear): ' num2str(mean(CaltechScoresLinearLOCERR))]);
disp([dataset ' mean LOC_ERR(LinearBilateralFiltered): ' num2str(mean(CaltechScoresLinearBilateralFilteredLOCERR))]);
disp([dataset ' mean LOC_ERR(Deformable): ' num2str(mean(CaltechScoresDeformableLOCERR))]);
disp([dataset ' mean LOC_ERR(DeformableBilateralFiltered): ' num2str(mean(CaltechScoresDeformableBilateralFilteredLOCERR))]);
disp([dataset ' mean LOC_ERR(TPS): ' num2str(mean(CaltechScoresTPSLOCERR))]);
disp([dataset ' mean LOC_ERR(Natural): ' num2str(mean(CaltechScoresNaturalLOCERR))]);



end

function [acc, iou, loc_err] = getPerformance(anno1, anno2, vx, vy)

[h1, w1] = size(anno1);
[h2, w2] = size(anno2);

[x1, y1] = meshgrid(1:w1, 1:h1);
x2 = x1+ vx;
y2 = y1+ vy;

% localization
[obj_x1, obj_y1] = ObjPtr(anno1);
[obj_x2, obj_y2] = ObjPtr(anno2);

tf = x2 >= 1 & x2 <= w2 & y2 >= 1 & y2 <= h2;
ptr_ind1 = sub2ind(size(anno1), y1(tf), x1(tf));
ptr_ind2 = sub2ind(size(anno2), y2(tf), x2(tf));
loc_err_map = inf(size(anno1));
loc_err_map(ptr_ind1) =...
    abs(obj_x1(ptr_ind1) - obj_x2(ptr_ind2)) + abs(obj_y1(ptr_ind1) - obj_y2(ptr_ind2));

% localization evaluation
[seg, in_bound] = TransferAnnotation(anno2, vx,vy);
% true_match = seg == anno1 & anno1 == 1 & in_bound;
% loc_err.correct_fg = mean2(loc_err_map(true_match));
% fg_match = seg == 1 & in_bound;
% loc_err.fg = mean2(loc_err_map(fg_match));
% loc_err.all = mean2(loc_err_map(in_bound));
loc_err = mean2(loc_err_map(in_bound));

% label transfer evluation
mean_acc = mean2(seg == anno1);
acc = mean_acc;
% fg = sum(seg(:) == anno1(:) & anno1(:) == 1)./sum(anno1(:));
% bg = sum(seg(:) == anno1(:) & anno1(:) == 0)./sum(anno1(:)==0);
%accuracy.fg = fg;
%accuracy.bg = bg;
i = seg == 1 & anno1 == 1;
u = seg == 1 | anno1 == 1;
iou = sum(i(:))/sum(u(:));



end


function [x1, y1] = ObjPtr(anno)

[y1, x1] = find(anno);
lx1 = min(x1);
rx1 = max(x1);
ty1 = min(y1);
dy1 = max(y1);
w1 = rx1-lx1 + 1;
h1 = dy1-ty1 + 1;
[x,y] = meshgrid(1:size(anno,2), 1:size(anno,1));
x1 = (x - lx1)./w1;
y1 = (y - ty1)./h1;

end

function [anno1, in_bound] = TransferAnnotation(anno2, vx,vy)
% x2 = x1 + vx, y2 = y1 + vy

[h1,w1] = size(vx);
[h2,w2] = size(anno2);

[x1,y1] = meshgrid(1:w1, 1:h1);
x2 = x1 + vx;
y2 = y1 + vy;

in_bound = x2 >= 1 & x2 <= w2 & y2 >= 1 & y2 <= h2;

inds1 = sub2ind([h1,w1], y1(in_bound), x1(in_bound));
inds2 = sub2ind([h2,w2], y2(in_bound), x2(in_bound));

anno1 = zeros(h1,w1);
anno1(inds1) = anno2(inds2);

end
