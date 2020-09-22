#include "mex.h"
#include <iostream>
#include <queue>
#include <vector>
#include <random>
#include <stdlib.h>     /* srand, rand */
#include <math.h>
#include <cfloat>
#include <set>
#include <chrono>
#include <omp.h>


/* NOTE: compile with mex -v -largeArrayDims CDEBUGFLAGS=" " COPTIMFLAGS="-O3" CXXDEBUGFLAGS=" " LDDEBUGFLAGS=" " LDCXXDEBUGFLAGS=" " LDCXXOPTIMFLAGS="-O3" LDOPTIMFLAGS="-O3" CXXOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -std=c++11 -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" ./computeHOGSimilarityMatrix.cpp in MATLAB */

using namespace std;
using namespace std::chrono;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    // data order:
    // box1Features [0]
    // box2Features [1]
    
    // get total number of cells for box1Features
    mwSize totalNumberOfBoxes1;
    totalNumberOfBoxes1 = mxGetNumberOfElements(prhs[0]);
    cout << "totalNumberOfBoxes1: " << totalNumberOfBoxes1 << endl;
    
    // get total number of cells for box2Features
    mwSize totalNumberOfBoxes2;
    totalNumberOfBoxes2 = mxGetNumberOfElements(prhs[1]);
    cout << "totalNumberOfBoxes2: " << totalNumberOfBoxes2 << endl;
        
    // store similarity matrix
    vector< vector<double> > hogSimilarityMatrix(totalNumberOfBoxes1, vector<double>(totalNumberOfBoxes2, 0.0));
    
    // get HOG and HOGBias field ids
    const mxArray* firstBoxPointer = mxGetCell(prhs[0], 0);
    unsigned int numberOfFields = mxGetNumberOfFields(firstBoxPointer);
    cout << "numberOfFields: " << numberOfFields << endl;
    const char* fieldName;
    string fieldNameString;
    unsigned int HOGFieldId;
    unsigned int HOGBiasFieldId;
    for (unsigned int i = 0; i < numberOfFields; i++)
    {
        fieldName = mxGetFieldNameByNumber(firstBoxPointer, i);
        fieldNameString = string(fieldName);
        if (fieldNameString.compare("HOG") == 0)
        {
            HOGFieldId = i;
            
        }
        else if (fieldNameString.compare("HOGBias") == 0)
        {
            HOGBiasFieldId = i;
        }      
    }
    cout << "HOGFieldId: " << HOGFieldId << endl;
    cout << "HOGBiasFieldId: " << HOGBiasFieldId << endl;
    
//     // compute similarities
//     auto start = system_clock::now();
//     for (unsigned int box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
//     {
//         const mxArray* hog1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, HOGFieldId);
//         double* hog1Features = (double*)mxGetData(hog1);
//         double hog1Bias = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, HOGBiasFieldId) ) )[0];
//         
//         
//         for (unsigned int box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
//         {
//             double sum = 0.0;
//             
//             const mxArray* hog2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, HOGFieldId);
//             double* hog2Features = (double*)mxGetData(hog2);
//             
//             // do dot product
//             const mwSize* hogDimensions = mxGetDimensions(hog1);
//             unsigned int featureDimension = (unsigned int)hogDimensions[0];
//             
//             for (unsigned int fDim = 0; fDim < featureDimension; fDim++)
//             {
//                 sum += hog1Features[fDim] * hog2Features[fDim];
//                 
//             }
//             
//             sum += hog1Bias;
//             
//             hogSimilarityMatrix.at(box1Id).at(box2Id) = max(sum, 0.0);
//             
//         }
//         
//         //cout << "Box1Id: " << box1Id << " vs. all done..." << endl;
//     }
//     auto end = system_clock::now();
//     auto elapsed = duration_cast<milliseconds>(end-start);
//     
//     cout << "Time elapsed: " << elapsed.count() << " milliseconds." << endl;
    
    
    
    // compute similarities (parallel implementation)
    
    
    unsigned int box1Id;
    unsigned int box2Id;
    const mxArray* hog1;
    double* hog1Features;
    double hog1Bias;
    double sum;
    const mxArray* hog2;
    double* hog2Features;
    const mwSize* hogDimensions;
    unsigned int featureDimension;
    unsigned int fDim;
        
    //omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());
   
    auto start = system_clock::now();
    #pragma omp parallel for default(none) \
                shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrix, HOGFieldId, HOGBiasFieldId, prhs, cout) \
                private(box1Id, box2Id, hog1, hog1Features, hog1Bias, sum, hog2, hog2Features, hogDimensions, featureDimension, fDim)
    for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
    {
        hog1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, HOGFieldId);
        hog1Features = (double*)mxGetData(hog1);
        hog1Bias = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, HOGBiasFieldId) ) )[0];
 
        for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)            
        {
            sum = 0.0;
            
            hog2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, HOGFieldId);
            hog2Features = (double*)mxGetData(hog2);
            
            // do dot product            
            hogDimensions = mxGetDimensions(hog1);
            featureDimension = (unsigned int)hogDimensions[0];
                        
            for (fDim = 0; fDim < featureDimension; fDim++)
            {
                sum += hog1Features[fDim] * hog2Features[fDim]; 
                
            }
            
            sum += hog1Bias;
            
            hogSimilarityMatrix.at(box1Id).at(box2Id) = max(sum, 0.0);

        }
        
        //cout << "Box1Id: " << box1Id << " vs. all done..." << endl;
    }
    auto end = system_clock::now();
    auto elapsed = duration_cast<milliseconds>(end-start);
    
    cout << "Time elapsed: " << elapsed.count() << " milliseconds." << endl;
    
        
    mwSize resultMatrixDims[2];
    resultMatrixDims[0] = totalNumberOfBoxes1;
    resultMatrixDims[1] = totalNumberOfBoxes2;
    plhs[0] = mxCreateNumericArray(2, resultMatrixDims, mxDOUBLE_CLASS, mxREAL);
    double* resultMatrixPointer = (double*)mxGetData(plhs[0]);
    
    for (unsigned int box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
    {
        for (unsigned int box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
        {
            resultMatrixPointer[totalNumberOfBoxes1 * box2Id + box1Id] = hogSimilarityMatrix.at(box1Id).at(box2Id);
        }
    }
    
    
}
