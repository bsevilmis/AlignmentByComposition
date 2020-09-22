function output = setAuxiliaryFilePaths()

dataset = 'TSS_CVPR2016'; %TSS_CVPR2016, PF-dataset, PF-dataset-PASCAL, Caltech101

mainDatasetFolder = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/Images/' dataset];

saliencyMapsFolder = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/SaliencyMaps/' dataset];

objectDetectionsFolder = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ObjectDetections/' dataset];

objectProposalsCodeFolder = '/users/bsevilmi/scratch/AlignmentByComposition/external/SelectiveSearchCodeIJCV';

vlFeatFolder = '/users/bsevilmi/scratch/AlignmentByComposition/external/vlfeat-0.9.17';

dsiftDictFile = '/users/bsevilmi/scratch/AlignmentByComposition/external/dsift_dict.mat';

hogFeaturesFolder = '/users/bsevilmi/scratch/AlignmentByComposition/external/feature';

imagePair1TextFileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/pair1.txt'];

imagePair2TextFileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/pair2.txt'];

deformableRegistrationFolder = '/users/bsevilmi/scratch/AlignmentByComposition/external/dsp-code';

resultsFolder = ['/users/bsevilmi/scratch/AlignmentByComposition/results/' dataset];

addpath(genpath(hogFeaturesFolder));
addpath(genpath(objectProposalsCodeFolder));
addpath(genpath(deformableRegistrationFolder));
addpath(genpath(vlFeatFolder));
vl_setup;

imagePair1TextFileIdentifier = fopen(imagePair1TextFileName, 'r');
pair1ImageNames = textscan(imagePair1TextFileIdentifier,'%s\n');
pair1ImageNames = pair1ImageNames{1};
fclose(imagePair1TextFileIdentifier);

imagePair2TextFileIdentifier = fopen(imagePair2TextFileName, 'r');
pair2ImageNames = textscan(imagePair2TextFileIdentifier,'%s\n');
pair2ImageNames = pair2ImageNames{1};
fclose(imagePair2TextFileIdentifier);

output.mainDatasetFolder = mainDatasetFolder;
output.saliencyMapsFolder = saliencyMapsFolder;
output.objectDetectionsFolder = objectDetectionsFolder;
output.objectProposalsCodeFolder = objectProposalsCodeFolder;
output.vlFeatFolder = vlFeatFolder;
output.dsiftDictFile = dsiftDictFile;
output.hogFeaturesFolder = hogFeaturesFolder;
output.pair1ImageNames = pair1ImageNames;
output.pair2ImageNames = pair2ImageNames;
output.resultsFolder = resultsFolder;

end
