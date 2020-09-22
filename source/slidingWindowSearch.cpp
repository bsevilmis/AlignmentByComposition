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
#include "string.h"


/* NOTE: compile with mex -v -largeArrayDims CDEBUGFLAGS=" " COPTIMFLAGS="-O3" CXXDEBUGFLAGS=" " LDDEBUGFLAGS=" " LDCXXDEBUGFLAGS=" " LDCXXOPTIMFLAGS="-O3" LDOPTIMFLAGS="-O3" CXXOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -std=c++11 -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" ./slidingWindowSearch.cpp in MATLAB */

using namespace std;
using namespace std::chrono;


// small value, used to avoid division by zero
#define eps 0.0001

// unit vectors used to compute gradient orientation
double uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
double vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};


// vector< vector< vector<double> > > subarray(unsigned char* image2Pointer, unsigned int rowSize, unsigned int colSize,
//         int candidateBoxY1, int candidateBoxY2,
//         int candidateBoxX1, int candidateBoxX2)
// {
//     
//     vector< vector< vector<double> > > window(candidateBoxY2 - candidateBoxY1 + 1, vector< vector<double> >(candidateBoxX2 - candidateBoxX1 + 1, vector<double>(3, 0.0)));
//     
//     for (int r = candidateBoxY1; r <= candidateBoxY2; r++)
//     {
//         
//         for (int c = candidateBoxX1; c <= candidateBoxX2; c++)
//         {
//             int correctedR = max(0,min(r, ((int)rowSize - 1)));
//             int correctedC = max(0,min(c, ((int)colSize - 1)));
//             
// 
//             window.at(r - candidateBoxY1).at(c - candidateBoxX1).at(0) = image2Pointer[ (correctedC) * rowSize + correctedR];
//             window.at(r - candidateBoxY1).at(c - candidateBoxX1).at(1) = image2Pointer[ colSize * rowSize + (correctedC) * rowSize + correctedR];
//             window.at(r - candidateBoxY1).at(c - candidateBoxX1).at(2) = image2Pointer[ 2 * colSize * rowSize + (correctedC) * rowSize + correctedR];
// 
//         }
//     }
//     
//     
//     return window;
//     
// }

vector<double> subarray(unsigned char* image2Pointer, unsigned int rowSize, unsigned int colSize, int windowRowSize, int windowColSize,
        int candidateBoxY1, int candidateBoxY2,
        int candidateBoxX1, int candidateBoxX2)
{
            
    vector<double> window(windowRowSize*windowColSize*3, 0.0);
    
    for (int r = candidateBoxY1; r <= candidateBoxY2; r++)
    {
        
        for (int c = candidateBoxX1; c <= candidateBoxX2; c++)
        {
            int correctedR = max(0,min(r, ((int)rowSize - 1)));
            int correctedC = max(0,min(c, ((int)colSize - 1)));
            
            window[ (c - candidateBoxX1) * windowRowSize +  (r - candidateBoxY1) ] = image2Pointer[ (correctedC) * rowSize + correctedR];
            window[ windowRowSize * windowColSize + (c - candidateBoxX1) * windowRowSize +  (r - candidateBoxY1)] = image2Pointer[ colSize * rowSize + (correctedC) * rowSize + correctedR];
            window[ 2 * windowRowSize * windowColSize + (c - candidateBoxX1) * windowRowSize +  (r - candidateBoxY1)] = image2Pointer[ 2 * colSize * rowSize + (correctedC) * rowSize + correctedR];
            
        }
    }
    
    
    return window;
    
}
        
    
// vector< vector< vector<double> > > resizeImage(vector< vector< vector<double> > > window, unsigned int cropSizeY, unsigned int cropSizeX)
// {
//     vector< vector< vector<double> > > patch(cropSizeY, vector< vector<double> >(cropSizeX, vector<double>(3, 0.0)));
//     int sourceHeight = window.size();
//     int sourceWidth = window.at(0).size();
//     
//     double ratioX = (double)cropSizeX / (double)sourceWidth;
//     double ratioY = (double)cropSizeY / (double)sourceHeight;
//     double x,y;
//     int xInt, yInt;
//     double dx, dy, s;
//     int u,v;
//     
//     for (unsigned int r = 0; r < cropSizeY; r++)
//     {
//         for (unsigned int c = 0; c < cropSizeX; c++)
//         {
//             x = ((double)c+1)/ratioX - 1;
//             y = ((double)r+1)/ratioY - 1;
//             
//             xInt = x;
//             yInt = y;
//             
//             dx = max(min(x-xInt,1.0),0.0);
//             dy = max(min(y-yInt,1.0),0.0);
//             
//             for (int m = 0; m<=1; m++)
//             {
//                 for (int n = 0; n<=1; n++)
//                 {
//                     u = min(max(xInt + m, 0), sourceWidth - 1);
//                     v = min(max(yInt + n, 0), sourceHeight - 1);
//                     
//                     s = fabs(1-m-dx) * fabs(1-n-dy);
//                     
//                     patch.at(r).at(c).at(0) = patch.at(r).at(c).at(0) + window.at(v).at(u).at(0) * s;
//                     patch.at(r).at(c).at(1) = patch.at(r).at(c).at(1) + window.at(v).at(u).at(1) * s;
//                     patch.at(r).at(c).at(2) = patch.at(r).at(c).at(2) + window.at(v).at(u).at(2) * s;
//                     
//                     
//                 }
//             }
//             
//          
//             
//         }
//     }
//     
//     
//     return patch;
//     
//     
// }

vector<double> resizeImage(vector<double> window, int windowRowSize, int windowColSize, unsigned int cropSizeY, unsigned int cropSizeX)
{
    vector<double> patch(cropSizeY * cropSizeX * 3, 0.0);
    int sourceHeight = windowRowSize;
    int sourceWidth = windowColSize;
    
    double ratioX = (double)cropSizeX / (double)sourceWidth;
    double ratioY = (double)cropSizeY / (double)sourceHeight;
    double x,y;
    int xInt, yInt;
    double dx, dy, s;
    int u,v;
    
    for (unsigned int r = 0; r < cropSizeY; r++)
    {
        for (unsigned int c = 0; c < cropSizeX; c++)
        {
            x = ((double)c+1)/ratioX - 1;
            y = ((double)r+1)/ratioY - 1;
            
            xInt = x;
            yInt = y;
            
            dx = max(min(x-xInt,1.0),0.0);
            dy = max(min(y-yInt,1.0),0.0);
            
            for (int m = 0; m<=1; m++)
            {
                for (int n = 0; n<=1; n++)
                {
                    u = min(max(xInt + m, 0), sourceWidth - 1);
                    v = min(max(yInt + n, 0), sourceHeight - 1);
                    
                    s = fabs(1-m-dx) * fabs(1-n-dy);
                    
                    
                    patch[ cropSizeY * c + r ] = patch[ cropSizeY * c + r ] + window[ u * sourceHeight + v ] * s;
                    patch[ cropSizeY * cropSizeX + cropSizeY * c + r ] = patch[ cropSizeY * cropSizeX + cropSizeY * c + r ] + window[ sourceHeight * sourceWidth + u * sourceHeight + v ] * s;
                    patch[ 2 * cropSizeY * cropSizeX + cropSizeY * c + r ] = patch[ 2 * cropSizeY * cropSizeX + cropSizeY * c + r ] + window[ 2 * sourceHeight * sourceWidth + u * sourceHeight + v ] * s;
                    
                    
                    
                }
            }
            
         
            
        }
    }
    
    
    return patch;
    
    
}
        


// vector<double> getHOGFeatures(vector< vector< vector<double> > > patch, unsigned int szCell)
// {
//   
//   // get dimensions of the patch
//   const int rowSize = patch.size();
//   const int colSize = patch.at(0).size();
//   const int channels = patch.at(0).at(0).size();
//   
//   
//   // lets pass the patch data to a 1D contiguous vector<double>
//   vector<double> patchContiguous(rowSize*colSize*channels, 0.0);
//   for (unsigned int ch = 0; ch < channels; ch++)
//   {
//       for (unsigned int c = 0; c < colSize; c++)
//       {
//           for (unsigned int r = 0; r < rowSize; r++)
//           {              
//               patchContiguous[ch*colSize*rowSize + c*rowSize + r] = patch.at(r).at(c).at(ch);              
//           }          
//       }      
//   }
//     
//   int dims[3];
//   dims[0] = rowSize;
//   dims[1] = colSize;
//   dims[2] = channels;
//   
//   int sbin = szCell;
//     
//   // memory for caching orientation histograms & their norms
//   int blocks[2];
//   blocks[0] = (int)round((double)dims[0]/(double)sbin);
//   blocks[1] = (int)round((double)dims[1]/(double)sbin);
//   
//   vector<double> histVector(blocks[0]*blocks[1]*18, 0.0);
//   vector<double> normVector(blocks[0]*blocks[1], 0.0);
//   
//   // memory for HOG features
//   int out[3];
//   out[0] = max(blocks[0]-2, 0);
//   out[1] = max(blocks[1]-2, 0);
//   out[2] = 27+4+1;
//   
//   vector<double> featVector(out[0]*out[1]*out[2], 0.0);
//   
//   // define pointers
//   double* im = patchContiguous.data();
//   double* hist = histVector.data();
//   double* norm = normVector.data();  
//   double* feat = featVector.data();
//   
//   
//   int visible[2];
//   visible[0] = blocks[0]*sbin;
//   visible[1] = blocks[1]*sbin;
//   
//   for (int x = 1; x < visible[1]-1; x++) {
//       for (int y = 1; y < visible[0]-1; y++) {
//           // first color channel
//           double *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
//           double dy = *(s+1) - *(s-1);
//           double dx = *(s+dims[0]) - *(s-dims[0]);
//           double v = dx*dx + dy*dy;
//           
//           // second color channel
//           s += dims[0]*dims[1];
//           double dy2 = *(s+1) - *(s-1);
//           double dx2 = *(s+dims[0]) - *(s-dims[0]);
//           double v2 = dx2*dx2 + dy2*dy2;
//           
//           // third color channel
//           s += dims[0]*dims[1];
//           double dy3 = *(s+1) - *(s-1);
//           double dx3 = *(s+dims[0]) - *(s-dims[0]);
//           double v3 = dx3*dx3 + dy3*dy3;
//           
//           // pick channel with strongest gradient
//           if (v2 > v) {
//               v = v2;
//               dx = dx2;
//               dy = dy2;
//           }
//           if (v3 > v) {
//               v = v3;
//               dx = dx3;
//               dy = dy3;
//           }
//           
//           // snap to one of 18 orientations
//           double best_dot = 0;
//           int best_o = 0;
//           for (int o = 0; o < 9; o++) {
//               double dot = uu[o]*dx + vv[o]*dy;
//               if (dot > best_dot) {
//                   best_dot = dot;
//                   best_o = o;
//               } else if (-dot > best_dot) {
//                   best_dot = -dot;
//                   best_o = o+9;
//               }
//           }
//           
//           // add to 4 histograms around pixel using linear interpolation
//           double xp = ((double)x+0.5)/(double)sbin - 0.5;
//           double yp = ((double)y+0.5)/(double)sbin - 0.5;
//           int ixp = (int)floor(xp);
//           int iyp = (int)floor(yp);
//           double vx0 = xp-ixp;
//           double vy0 = yp-iyp;
//           double vx1 = 1.0-vx0;
//           double vy1 = 1.0-vy0;
//           v = sqrt(v);
//           
//           if (ixp >= 0 && iyp >= 0) {
//               *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
//                       vx1*vy1*v;
//           }
//           
//           if (ixp+1 < blocks[1] && iyp >= 0) {
//               *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
//                       vx0*vy1*v;
//           }
//           
//           if (ixp >= 0 && iyp+1 < blocks[0]) {
//               *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
//                       vx1*vy0*v;
//           }
//           
//           if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
//               *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
//                       vx0*vy0*v;
//           }
//       }
//   }
//   
//   // compute energy in each block by summing over orientations
//   for (int o = 0; o < 9; o++) {
//       double *src1 = hist + o*blocks[0]*blocks[1];
//       double *src2 = hist + (o+9)*blocks[0]*blocks[1];
//       double *dst = norm;
//       double *end = norm + blocks[1]*blocks[0];
//       while (dst < end) {
//           *(dst++) += (*src1 + *src2) * (*src1 + *src2);
//           src1++;
//           src2++;
//       }
//   }
//   
//   // compute features
//   for (int x = 0; x < out[1]; x++) {
//       for (int y = 0; y < out[0]; y++) {
//           double *dst = feat + x*out[0] + y;
//           double *src, *p, n1, n2, n3, n4;
//           
//           p = norm + (x+1)*blocks[0] + y+1;
//           n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//           p = norm + (x+1)*blocks[0] + y;
//           n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//           p = norm + x*blocks[0] + y+1;
//           n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//           p = norm + x*blocks[0] + y;
//           n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
//           double t1 = 0;
//           double t2 = 0;
//           double t3 = 0;
//           double t4 = 0;
//           double bh_test=0;
//           // contrast-sensitive features
//           src = hist + (x+1)*blocks[0] + (y+1);
//           for (int o = 0; o < 18; o++) {
//               bh_test=bh_test+(*src * n1)+(*src * n2)+(*src * n3)+(*src * n4);
//               double h1 = min(*src * n1, 0.2);
//               double h2 = min(*src * n2, 0.2);
//               double h3 = min(*src * n3, 0.2);
//               double h4 = min(*src * n4, 0.2);
//               *dst = 0.5 * (h1 + h2 + h3 + h4);
//               t1 += h1;
//               t2 += h2;
//               t3 += h3;
//               t4 += h4;
//               dst += out[0]*out[1];
//               src += blocks[0]*blocks[1];
//           }
//           //printf("%f\n", bh_test);
//           // contrast-insensitive features
//           src = hist + (x+1)*blocks[0] + (y+1);
//           for (int o = 0; o < 9; o++) {
//               double sum = *src + *(src + 9*blocks[0]*blocks[1]);
//               double h1 = min(sum * n1, 0.2);
//               double h2 = min(sum * n2, 0.2);
//               double h3 = min(sum * n3, 0.2);
//               double h4 = min(sum * n4, 0.2);
//               *dst = 0.5 * (h1 + h2 + h3 + h4);
//               dst += out[0]*out[1];
//               src += blocks[0]*blocks[1];
//           }
//           
//           // texture features
//           *dst = 0.2357 * t1;
//           dst += out[0]*out[1];
//           *dst = 0.2357 * t2;
//           dst += out[0]*out[1];
//           *dst = 0.2357 * t3;
//           dst += out[0]*out[1];
//           *dst = 0.2357 * t4;
//           
//           // truncation feature
//           dst += out[0]*out[1];
//           *dst = 0;
//       }
//   }
//   
// 
//   
//   // return the result
//   vector<double> HOGFeatures(out[0]*out[1]*(out[2]-1), 0.0);
//   
//   HOGFeatures.assign(featVector.begin(), featVector.end() - (out[0]*out[1]));
//   
//   
//   return HOGFeatures;
//   
// }

vector<double> getHOGFeatures(vector<double> patch, unsigned int patchRowSize, unsigned int patchColSize, unsigned int szCell)
{
      
  int dims[3];
  dims[0] = patchRowSize;
  dims[1] = patchColSize;
  dims[2] = 3;
  
  int sbin = szCell;
    
  // memory for caching orientation histograms & their norms
  int blocks[2];
  blocks[0] = (int)round((double)dims[0]/(double)sbin);
  blocks[1] = (int)round((double)dims[1]/(double)sbin);
  
  vector<double> histVector(blocks[0]*blocks[1]*18, 0.0);
  vector<double> normVector(blocks[0]*blocks[1], 0.0);
  
  // memory for HOG features
  int out[3];
  out[0] = max(blocks[0]-2, 0);
  out[1] = max(blocks[1]-2, 0);
  out[2] = 27+4+1;
  
  vector<double> featVector(out[0]*out[1]*out[2], 0.0);
  
  // define pointers
  double* im = patch.data();
  double* hist = histVector.data();
  double* norm = normVector.data();  
  double* feat = featVector.data();
  
  
  int visible[2];
  visible[0] = blocks[0]*sbin;
  visible[1] = blocks[1]*sbin;
  
  for (int x = 1; x < visible[1]-1; x++) {
      for (int y = 1; y < visible[0]-1; y++) {
          // first color channel
          double *s = im + min(x, dims[1]-2)*dims[0] + min(y, dims[0]-2);
          double dy = *(s+1) - *(s-1);
          double dx = *(s+dims[0]) - *(s-dims[0]);
          double v = dx*dx + dy*dy;
          
          // second color channel
          s += dims[0]*dims[1];
          double dy2 = *(s+1) - *(s-1);
          double dx2 = *(s+dims[0]) - *(s-dims[0]);
          double v2 = dx2*dx2 + dy2*dy2;
          
          // third color channel
          s += dims[0]*dims[1];
          double dy3 = *(s+1) - *(s-1);
          double dx3 = *(s+dims[0]) - *(s-dims[0]);
          double v3 = dx3*dx3 + dy3*dy3;
          
          // pick channel with strongest gradient
          if (v2 > v) {
              v = v2;
              dx = dx2;
              dy = dy2;
          }
          if (v3 > v) {
              v = v3;
              dx = dx3;
              dy = dy3;
          }
          
          // snap to one of 18 orientations
          double best_dot = 0;
          int best_o = 0;
          for (int o = 0; o < 9; o++) {
              double dot = uu[o]*dx + vv[o]*dy;
              if (dot > best_dot) {
                  best_dot = dot;
                  best_o = o;
              } else if (-dot > best_dot) {
                  best_dot = -dot;
                  best_o = o+9;
              }
          }
          
          // add to 4 histograms around pixel using linear interpolation
          double xp = ((double)x+0.5)/(double)sbin - 0.5;
          double yp = ((double)y+0.5)/(double)sbin - 0.5;
          int ixp = (int)floor(xp);
          int iyp = (int)floor(yp);
          double vx0 = xp-ixp;
          double vy0 = yp-iyp;
          double vx1 = 1.0-vx0;
          double vy1 = 1.0-vy0;
          v = sqrt(v);
          
          if (ixp >= 0 && iyp >= 0) {
              *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
                      vx1*vy1*v;
          }
          
          if (ixp+1 < blocks[1] && iyp >= 0) {
              *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) +=
                      vx0*vy1*v;
          }
          
          if (ixp >= 0 && iyp+1 < blocks[0]) {
              *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
                      vx1*vy0*v;
          }
          
          if (ixp+1 < blocks[1] && iyp+1 < blocks[0]) {
              *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) +=
                      vx0*vy0*v;
          }
      }
  }
  
  // compute energy in each block by summing over orientations
  for (int o = 0; o < 9; o++) {
      double *src1 = hist + o*blocks[0]*blocks[1];
      double *src2 = hist + (o+9)*blocks[0]*blocks[1];
      double *dst = norm;
      double *end = norm + blocks[1]*blocks[0];
      while (dst < end) {
          *(dst++) += (*src1 + *src2) * (*src1 + *src2);
          src1++;
          src2++;
      }
  }
  
  // compute features
  for (int x = 0; x < out[1]; x++) {
      for (int y = 0; y < out[0]; y++) {
          double *dst = feat + x*out[0] + y;
          double *src, *p, n1, n2, n3, n4;
          
          p = norm + (x+1)*blocks[0] + y+1;
          n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
          p = norm + (x+1)*blocks[0] + y;
          n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
          p = norm + x*blocks[0] + y+1;
          n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
          p = norm + x*blocks[0] + y;
          n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
          double t1 = 0;
          double t2 = 0;
          double t3 = 0;
          double t4 = 0;
          double bh_test=0;
          // contrast-sensitive features
          src = hist + (x+1)*blocks[0] + (y+1);
          for (int o = 0; o < 18; o++) {
              bh_test=bh_test+(*src * n1)+(*src * n2)+(*src * n3)+(*src * n4);
              double h1 = min(*src * n1, 0.2);
              double h2 = min(*src * n2, 0.2);
              double h3 = min(*src * n3, 0.2);
              double h4 = min(*src * n4, 0.2);
              *dst = 0.5 * (h1 + h2 + h3 + h4);
              t1 += h1;
              t2 += h2;
              t3 += h3;
              t4 += h4;
              dst += out[0]*out[1];
              src += blocks[0]*blocks[1];
          }
          //printf("%f\n", bh_test);
          // contrast-insensitive features
          src = hist + (x+1)*blocks[0] + (y+1);
          for (int o = 0; o < 9; o++) {
              double sum = *src + *(src + 9*blocks[0]*blocks[1]);
              double h1 = min(sum * n1, 0.2);
              double h2 = min(sum * n2, 0.2);
              double h3 = min(sum * n3, 0.2);
              double h4 = min(sum * n4, 0.2);
              *dst = 0.5 * (h1 + h2 + h3 + h4);
              dst += out[0]*out[1];
              src += blocks[0]*blocks[1];
          }
          
          // texture features
          *dst = 0.2357 * t1;
          dst += out[0]*out[1];
          *dst = 0.2357 * t2;
          dst += out[0]*out[1];
          *dst = 0.2357 * t3;
          dst += out[0]*out[1];
          *dst = 0.2357 * t4;
          
          // truncation feature
          dst += out[0]*out[1];
          *dst = 0;
      }
  }
  

  
  // return the result
  vector<double> HOGFeatures(out[0]*out[1]*(out[2]-1), 0.0);
  
  HOGFeatures.assign(featVector.begin(), featVector.end() - (out[0]*out[1]));
  
  
  return HOGFeatures;
  
}


                        
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    // data order:
    // parts1 [0]
    // image2 [1]
    // slidingWindowSearchParams [2]
    
    
    // get total number of cells in parts1
    mwSize totalNumberOfPartsInImage1;
//    totalNumberOfPartsInImage1 = mxGetNumberOfElements(prhs[0]);
//     cout << "totalNumberOfPartsInImage1: " << totalNumberOfPartsInImage1 << endl;
    const mxArray* parts1StructPointer = mxGetCell(prhs[0], 0);
    unsigned int numberOfFields = mxGetNumberOfFields(parts1StructPointer);
    
    // get image2 matrix
    unsigned char* image2Pointer = (unsigned char*)mxGetData(prhs[1]);
    unsigned int rowSize = ( mxGetDimensions(prhs[1]) )[0];
    unsigned int colSize = ( mxGetDimensions(prhs[1]) )[1];
//     cout << "rowSize: " << rowSize << endl;
//     cout << "colSize: " << colSize << endl;
//     
//     
//     cout <<"image2(10,20,1): " << (unsigned int)image2Pointer[ (20-1) * rowSize + (10-1) ] << endl;
//     cout <<"image2(10,20,2): " << (unsigned int)image2Pointer[ colSize*rowSize + (20-1) * rowSize + (10-1) ] << endl;
//     cout <<"image2(10,20,3): " << (unsigned int)image2Pointer[ 2 * colSize * rowSize + (20-1) * rowSize + (10-1) ] << endl;
    
    
    // get field ids
    unsigned int featuresFieldId;
    unsigned int biasFieldId;
    const char* fieldName;
    string fieldNameString;
    
    for (unsigned int i = 0; i < numberOfFields; i++)
    {
        fieldName = mxGetFieldNameByNumber(parts1StructPointer, i);
        fieldNameString = string(fieldName);
        
        if (fieldNameString.compare("features") == 0)
        {
            featuresFieldId = i;
        }
        else if (fieldNameString.compare("bias") == 0)
        {
            biasFieldId = i;
        }
    }
    
    
    
    double* parts1FeaturesPointer = (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0],0),0,featuresFieldId) );
    double bias = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0],0),0,biasFieldId) ) )[0];
            
//     cout << "bias: " << bias << endl;
               
//     cout << "featuresFieldId: " << featuresFieldId << endl;
//     cout << "biasFieldId: " << biasFieldId << endl;
    
    const mxArray* slidingWindowSearchParamsStructPointer = prhs[2];
    numberOfFields = mxGetNumberOfFields(slidingWindowSearchParamsStructPointer);
//     cout << "numberOfFields: " << numberOfFields << endl;
    
    unsigned int strideFieldId;
    unsigned int widthPart2FieldId;
    unsigned int heightPart2FieldId;
    unsigned int boundingBox2FieldId;
    
    for (unsigned int i = 0; i < numberOfFields; i++)
    {
        fieldName = mxGetFieldNameByNumber(slidingWindowSearchParamsStructPointer, i);
        fieldNameString = string(fieldName);
        
        if (fieldNameString.compare("stride") == 0)
        {
            strideFieldId = i;
        }
        else if (fieldNameString.compare("widthPart2") == 0)
        {
            widthPart2FieldId = i;
        }
        else if (fieldNameString.compare("heightPart2") == 0)
        {
            heightPart2FieldId = i;
        }
        else if (fieldNameString.compare("boundingBox2") == 0)
        {
            boundingBox2FieldId = i;
        }
    }
    
//     cout << "strideFieldId: " << strideFieldId << endl;
//     cout << "widthPart2FieldId: " << widthPart2FieldId << endl;
//     cout << "heightPart2FieldId: " << heightPart2FieldId << endl;
    unsigned int stride = (unsigned int)( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,strideFieldId) ) )[0];
    double widthPart2 = ( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,widthPart2FieldId) ) )[0];
    double heightPart2 = ( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,heightPart2FieldId) ) )[0];
    unsigned int boundingBox2TopLeftY = (unsigned int)( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,boundingBox2FieldId) ) )[0]; 
    boundingBox2TopLeftY = boundingBox2TopLeftY - 1; //MATLAB to C++ conversion   
    unsigned int boundingBox2TopLeftX = (unsigned int)( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,boundingBox2FieldId) ) )[1];
    boundingBox2TopLeftX = boundingBox2TopLeftX - 1; //MATLAB to C++ conversion   
    unsigned int boundingBox2BottomRightY = (unsigned int)( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,boundingBox2FieldId) ) )[2];
    boundingBox2BottomRightY = boundingBox2BottomRightY - 1; //MATLAB to C++ conversion   
    unsigned int boundingBox2BottomRightX = (unsigned int)( (double*)mxGetData( mxGetFieldByNumber(prhs[2],0,boundingBox2FieldId) ) )[3]; 
    boundingBox2BottomRightX = boundingBox2BottomRightX - 1; //MATLAB to C++ conversion 

//     cout << "stride: " << stride << endl;
//     cout << "widthPart2: " << widthPart2 << endl;
//     cout << "heightPart2: " << heightPart2 << endl;
//     cout << "boundingBox2TopLeftY: " << boundingBox2TopLeftY << endl;
//     cout << "boundingBox2TopLeftX: " << boundingBox2TopLeftX << endl;
//     cout << "boundingBox2BottomRightY: " << boundingBox2BottomRightY << endl;
//     cout << "boundingBox2BottomRightX: " << boundingBox2BottomRightX << endl;
    
            
    // create heatMap
    vector<double> heatMap(rowSize * colSize, 0.0);
        
    // HOG Parameters
    unsigned int szCell = 8;
    unsigned int nX = 8;
    unsigned int nY = 8;
    unsigned int pixelsY = nY * szCell;
    unsigned int pixelsX = nX * szCell;
    unsigned int cropSizeY = (nY+2) * szCell;
    unsigned int cropSizeX = (nX+2) * szCell;
    
    int candidateBoxTopLeftY;
    int candidateBoxTopLeftX;
    int candidateBoxBottomRightY;
    int candidateBoxBottomRightX;
    unsigned int candidateBoxWidth;
    unsigned int candidateBoxHeight;
    double candidateBoxPadY;
    double candidateBoxPadX;
    int candidateBoxY1;
    int candidateBoxX1;
    int candidateBoxY2;
    int candidateBoxX2;    
    int windowRowSize;
    int windowColSize;
    
    vector<double> window;
    vector<double> patch;
    vector<double> hog;
    


//     for ( unsigned int r = 0; r < rowSize; r++)
//     {
//         for ( unsigned int c = 0; c < colSize; c++)
//         {
//             if ( c % stride != 0 || r % stride != 0 )
//             {
//                 continue;
//             }
//             
//             
//             if ( r >= boundingBox2TopLeftY && r <= boundingBox2BottomRightY &&
//                     c >= boundingBox2TopLeftX && c <= boundingBox2BottomRightX )
//             {
//                 
//                 
//                 candidateBoxTopLeftY = max(0.0, min((double)r - (double)heightPart2 * 0.5, ((double)rowSize-1)));
//                 candidateBoxTopLeftX = max(0.0, min((double)c - (double)widthPart2 * 0.5, ((double)colSize-1)));
//                 candidateBoxBottomRightY = max(0.0, min((double)r + (double)heightPart2 * 0.5, ((double)rowSize-1)));
//                 candidateBoxBottomRightX = max(0.0, min((double)c + (double)widthPart2 * 0.5, ((double)colSize-1)));
//                 
//                 
//                 candidateBoxWidth = candidateBoxBottomRightX - candidateBoxTopLeftX + 1;
//                 candidateBoxHeight = candidateBoxBottomRightY - candidateBoxTopLeftY + 1;
//                 
//                 candidateBoxPadY = ((double)szCell * candidateBoxHeight) / pixelsY;
//                 candidateBoxPadX = ((double)szCell * candidateBoxWidth) / pixelsX;
//                 
//                 candidateBoxY1 = round((double)candidateBoxTopLeftY - candidateBoxPadY);
//                 candidateBoxX1 = round((double)candidateBoxTopLeftX - candidateBoxPadX);
//                 candidateBoxY2 = round((double)candidateBoxBottomRightY + candidateBoxPadY);
//                 candidateBoxX2 = round((double)candidateBoxBottomRightX + candidateBoxPadX);
//                                 
//                 windowRowSize = (candidateBoxY2 - candidateBoxY1 + 1);
//                 windowColSize = (candidateBoxX2 - candidateBoxX1 + 1);
//                 
//                 window = subarray(image2Pointer, rowSize, colSize, windowRowSize, windowColSize, candidateBoxY1, candidateBoxY2, candidateBoxX1, candidateBoxX2);
//                 patch = resizeImage(window, windowRowSize, windowColSize, cropSizeY, cropSizeX);
// 
//                 
//                 hog = getHOGFeatures(patch, cropSizeY, cropSizeX, szCell);
//                 
//                 // dot product
//                 for (unsigned int featDim = 0; featDim < hog.size(); featDim++)
//                 {
//                     heatMap[ c*rowSize + r ] = heatMap[ c*rowSize + r ] + hog.at(featDim) * parts1FeaturesPointer[featDim]; 
//                 }
//                 
//                 heatMap[ c*rowSize + r ] = heatMap[ c*rowSize + r ] + bias;
//                 
//                 heatMap[ c*rowSize + r ] = max(0.0, heatMap[ c*rowSize + r ]);
//                                                                              
//             }
//             
//         }
//         
//         cout << "r: " << r << " done..." << endl;
//     }
    
    
    // PARALLEL IMPLEMENTATION
    
    //omp_set_num_threads(10);
    omp_set_num_threads(omp_get_max_threads());
    
    unsigned int r;
    unsigned int c;
    unsigned int featDim;
    
    
    #pragma omp parallel for default(none) \
        shared(rowSize, colSize, stride, boundingBox2TopLeftY, boundingBox2BottomRightY, boundingBox2TopLeftX, boundingBox2BottomRightX, \
                heightPart2, widthPart2, szCell, pixelsY, pixelsX, image2Pointer, cropSizeY, cropSizeX, heatMap, parts1FeaturesPointer, bias, cout) \
                        private(r, c, candidateBoxTopLeftY, candidateBoxTopLeftX, candidateBoxBottomRightY, candidateBoxBottomRightX, \
                                candidateBoxWidth, candidateBoxHeight, candidateBoxPadY, candidateBoxPadX, candidateBoxY1, candidateBoxX1, \
                                candidateBoxY2, candidateBoxX2, windowRowSize, windowColSize, window, patch, hog, featDim)                                                                      
    for (r = 0; r < rowSize; r++)
    {
        for (c = 0; c < colSize; c++)
        {
            if ( c % stride != 0 || r % stride != 0 )
            {
                continue;
            }
            
            
            if ( r >= boundingBox2TopLeftY && r <= boundingBox2BottomRightY &&
                    c >= boundingBox2TopLeftX && c <= boundingBox2BottomRightX )
            {
                
                
                candidateBoxTopLeftY = max(0.0, min((double)r - (double)heightPart2 * 0.5, ((double)rowSize-1)));
                candidateBoxTopLeftX = max(0.0, min((double)c - (double)widthPart2 * 0.5, ((double)colSize-1)));
                candidateBoxBottomRightY = max(0.0, min((double)r + (double)heightPart2 * 0.5, ((double)rowSize-1)));
                candidateBoxBottomRightX = max(0.0, min((double)c + (double)widthPart2 * 0.5, ((double)colSize-1)));
                
                
                candidateBoxWidth = candidateBoxBottomRightX - candidateBoxTopLeftX + 1;
                candidateBoxHeight = candidateBoxBottomRightY - candidateBoxTopLeftY + 1;
                
                candidateBoxPadY = ((double)szCell * candidateBoxHeight) / pixelsY;
                candidateBoxPadX = ((double)szCell * candidateBoxWidth) / pixelsX;
                
                candidateBoxY1 = round((double)candidateBoxTopLeftY - candidateBoxPadY);
                candidateBoxX1 = round((double)candidateBoxTopLeftX - candidateBoxPadX);
                candidateBoxY2 = round((double)candidateBoxBottomRightY + candidateBoxPadY);
                candidateBoxX2 = round((double)candidateBoxBottomRightX + candidateBoxPadX);
                                
                windowRowSize = (candidateBoxY2 - candidateBoxY1 + 1);
                windowColSize = (candidateBoxX2 - candidateBoxX1 + 1);
                
                window = subarray(image2Pointer, rowSize, colSize, windowRowSize, windowColSize, candidateBoxY1, candidateBoxY2, candidateBoxX1, candidateBoxX2);
                patch = resizeImage(window, windowRowSize, windowColSize, cropSizeY, cropSizeX);

                
                hog = getHOGFeatures(patch, cropSizeY, cropSizeX, szCell);
                
                // dot product
                for (featDim = 0; featDim < hog.size(); featDim++)
                {
                    heatMap[ c*rowSize + r ] = heatMap[ c*rowSize + r ] + hog.at(featDim) * parts1FeaturesPointer[featDim]; 
                }
                
                heatMap[ c*rowSize + r ] = heatMap[ c*rowSize + r ] + bias;
                
                heatMap[ c*rowSize + r ] = max(0.0, heatMap[ c*rowSize + r ]);
                                                                             
            }
            
        }
        
       // cout << "r: " << r << " done..." << endl;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    mwSize outDims[2];
    outDims[0] = rowSize;
    outDims[1] = colSize;
    plhs[0] = mxCreateNumericArray(2, outDims, mxDOUBLE_CLASS, mxREAL);
    double* outputPointer = (double*)mxGetPr(plhs[0]);
    copy(heatMap.begin(), heatMap.end(), outputPointer);
    
    mwSize outDims2[3];
    outDims2[0] = windowRowSize;
    outDims2[1] = windowColSize;
    outDims2[2] = 3;
    plhs[1] = mxCreateNumericArray(3, outDims2, mxDOUBLE_CLASS, mxREAL);
    double* outputPointer2 = (double*)mxGetPr(plhs[1]);
    copy(window.begin(), window.end(), outputPointer2);
    
    
    
    return;
    
    
    
//     
//     
//     
//     
//     
//     // box1Features [0]
//     // box2Features [1]
//     // hogSimilarityMatrix [2]
//     // spmSimilarityMatrix [3]
//     
//     // get total number of cells for box1Features
//     mwSize totalNumberOfBoxes1;
//     totalNumberOfBoxes1 = mxGetNumberOfElements(prhs[0]);
//     cout << "totalNumberOfBoxes1: " << totalNumberOfBoxes1 << endl;
//     
//     // get total number of cells for box2Features
//     mwSize totalNumberOfBoxes2;
//     totalNumberOfBoxes2 = mxGetNumberOfElements(prhs[1]);
//     cout << "totalNumberOfBoxes2: " << totalNumberOfBoxes2 << endl;
//     
//     // get hogSimilarityMatrixPointer
//     double* hogSimilarityMatrixPointer = (double*)mxGetData(prhs[2]);
//     
// //     // get spmSimilarityMatrixPointer
// //     double* spmSimilarityMatrixPointer = (double*)mxGetData(prhs[3]);
//         
//     // store similarity matrix
//     vector< vector<double> > similarityMatrix(totalNumberOfBoxes1, vector<double>(totalNumberOfBoxes2, 0.0));
//     
//     // get field ids
//     
//     // saliency
//     // detectionClass
//     // detectionClassJaccardIndex
//     // SPM
//     // largerBoxIds
//     
// 
//     const mxArray* firstBoxPointer = mxGetCell(prhs[0], 0);
//     unsigned int numberOfFields = mxGetNumberOfFields(firstBoxPointer);
//     cout << "numberOfFields: " << numberOfFields << endl;
//     const char* fieldName;
//     string fieldNameString;
//     
//     unsigned int saliencyFieldId;
//     unsigned int detectionClassFieldId;
//     unsigned int detectionClassJaccardIndexFieldId;
// //     unsigned int SPMFieldId;
//     unsigned int largerBoxIdsFieldId;
// //     unsigned int skipSPMFieldId;
//     
// 
//     for (unsigned int i = 0; i < numberOfFields; i++)
//     {
//         fieldName = mxGetFieldNameByNumber(firstBoxPointer, i);
//         fieldNameString = string(fieldName);
//         
//         if (fieldNameString.compare("saliency") == 0)
//         {
//             saliencyFieldId = i;            
//         }
//         else if (fieldNameString.compare("detectionClass") == 0)
//         {
//             detectionClassFieldId = i;
//         }
//         else if (fieldNameString.compare("detectionClassJaccardIndex") == 0)
//         {
//             detectionClassJaccardIndexFieldId = i;
//         }
// //         else if (fieldNameString.compare("SPM") == 0)
// //         {
// //             SPMFieldId = i;
// //         }
//         else if (fieldNameString.compare("largerBoxIds") == 0)
//         {
//             largerBoxIdsFieldId = i;
//         }
// //         else if (fieldNameString.compare("skipSPM") == 0)
// //         {
// //             skipSPMFieldId = i;
// //         }
//     }
//     cout << "saliencyFieldId: " << saliencyFieldId << endl;
//     cout << "detectionClassFieldId: " << detectionClassFieldId << endl;
//     cout << "detectionClassJaccardIndexFieldId: " << detectionClassJaccardIndexFieldId << endl;
// //     cout << "SPMFieldId: " << SPMFieldId << endl;
//     cout << "largerBoxIdsFieldId: " << largerBoxIdsFieldId << endl;
// //     cout << "skipSPMFieldId: " << skipSPMFieldId << endl;
//     
//     // first find maximum HOG match response for every box in image 1
//     vector<double> maxHOGMatchScores(totalNumberOfBoxes1, 0.0);    
//     
//     unsigned int box1Id;
//     unsigned int box2Id;
//     double currentScore;
//     
//     omp_set_num_threads(4);
//         
//     #pragma omp parallel for default(none) \
//         shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, maxHOGMatchScores) \
//                 private(box1Id, box2Id, currentScore)
//     for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
//     {
//         for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
//         {
//             currentScore = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id];
//             if( currentScore > maxHOGMatchScores.at(box1Id) )
//             {
//                 maxHOGMatchScores.at(box1Id) = currentScore; 
//             }            
//         }        
//     }
//     
// //     // first find maximum SPM match response for every box in image 1
// //     vector<double> maxSPMMatchScores(totalNumberOfBoxes1, 0.0);
// //     
// //     omp_set_num_threads(4);
// //     
// //     #pragma omp parallel for default(none) \
// //         shared(totalNumberOfBoxes1, totalNumberOfBoxes2, spmSimilarityMatrixPointer, maxSPMMatchScores) \
// //                 private(box1Id, box2Id, currentScore)
// //     for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
// //     {
// //         for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
// //         {
// //             currentScore = spmSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id];
// //             if( currentScore > maxSPMMatchScores.at(box1Id) )
// //             {
// //                 maxSPMMatchScores.at(box1Id) = currentScore;
// //             }
// //         }
// //     }
//             
// //     // compute similarities
// //         
// //     auto start = system_clock::now();
// //     for (unsigned int box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
// //     {
// //         
// //         double saliency1 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, saliencyFieldId) ) )[0];
// //         
// //         const mxArray* detectionClass1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassFieldId);        
// //         
// // //         char* buf1;
// // //         size_t buflen1 = mxGetN(detectionClass1) * sizeof(mxChar) + 1;
// // //         buf1 = (char*)mxMalloc(buflen1);
// // //         mxGetString(detectionClass1, buf1, (mwSize)buflen1);
// //         
// //         size_t buflen1 = mxGetN(detectionClass1);
// //         string detectionClass1String(buflen1, ' ');
// //         for (unsigned stringIterator1 = 0; stringIterator1 < buflen1; stringIterator1++)
// //         {
// //             detectionClass1String[stringIterator1] = ((char*)mxGetData(detectionClass1))[0];
// //         }
// //         
// // 
// //         double detectionClass1JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
// //         
// //         const mxArray* spm1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, SPMFieldId);
// //         double* spm1Features = (double*)mxGetData(spm1);
// //         
// //         const mxArray* largerBoxIds1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, largerBoxIdsFieldId);
// //         
// //                 
// //         for (unsigned int box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
// //         {
// //             
// //             double saliency2 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, saliencyFieldId) ) )[0];
// //             
// //             const mxArray* detectionClass2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassFieldId);
// //          
// // //             char* buf2;
// // //             size_t buflen2 = mxGetN(detectionClass2) * sizeof(mxChar) + 1;
// // //             buf2 = (char*)mxMalloc(buflen2);
// // //             mxGetString(detectionClass2, buf2, (mwSize)buflen2);
// //             
// //             size_t buflen2 = mxGetN(detectionClass2);
// //             string detectionClass2String(buflen2, ' ');
// //             for (unsigned stringIterator2 = 0; stringIterator2 < buflen2; stringIterator2++)
// //             {
// //                 detectionClass2String[stringIterator2] = ((char*)mxGetData(detectionClass2))[0];
// //             }
// //                         
// //             double detectionClass2JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
// //             
// //             const mxArray* spm2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, SPMFieldId);
// //             double* spm2Features = (double*)mxGetData(spm2);
// //                         
// //             // add saliency scores
// //             similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (saliency1 + saliency2);
// //             
// //             // add detection scores
// // //             if( !(strcmp(buf1, "background") == 0 || strcmp(buf2, "background") == 0) )
// // //             {
// // //                 if ( strcmp(buf1, buf2) == 0 )
// // //                 {
// // //                     similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (detectionClass1JaccardIndex + detectionClass2JaccardIndex);
// // //                 }
// // //             }
// //             if( !(detectionClass1String.compare("background") == 0 || detectionClass2String.compare("background") == 0) )
// //             {
// //                 if ( detectionClass1String.compare(detectionClass2String) == 0 )
// //                 {
// //                     similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (detectionClass1JaccardIndex + detectionClass2JaccardIndex);
// //                 }
// //             }
// //             
// //             // add similarity between SPMs
// //             const mwSize* spmDimensions = mxGetDimensions(spm1);
// //             unsigned featureDimension = (unsigned int)spmDimensions[1];
// //             
// //             double spmDistance = 0;
// //             for (unsigned int fDim = 0; fDim < featureDimension; fDim++)
// //             {
// //                 spmDistance +=  ( (spm1Features[fDim] - spm2Features[fDim]) * (spm1Features[fDim] - spm2Features[fDim]) ) /
// //                         ( spm1Features[fDim] + spm2Features[fDim] + DBL_EPSILON);                
// //             }
// //             spmDistance  = 0.5 * spmDistance;
// //             double spmSimilarity = 1.0 - spmDistance;
// //             
// //             similarityMatrix.at(box1Id).at(box2Id) += spmSimilarity;
// //             
// //             
// //             // add similarity between HOGs
// //             double contrastHOG = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
// //                     maxOuterContrast(hogSimilarityMatrixPointer, totalNumberOfBoxes1, totalNumberOfBoxes2, largerBoxIds1);
// //             
// //             similarityMatrix.at(box1Id).at(box2Id) += contrastHOG;
// //             
// // //             mxFree(buf2);
// //         }
// //         
// // //         mxFree(buf1);
// //         //cout << "Box1Id: " << box1Id << " vs. all done..." << endl;
// //     }
// //     auto end = system_clock::now();
// //     auto elapsed = duration_cast<milliseconds>(end-start);
// //     
// //     cout << "Time elapsed: " << elapsed.count() << " milliseconds." << endl;
//     
//     
//     
//     // compute similarities (parallel implementation)
//     
//     
//     //unsigned int box1Id;
//     double saliency1;
//     const mxArray* detectionClass1;
//     size_t buflen1;
//     string detectionClass1String;
//     unsigned stringIterator1;
//     double detectionClass1JaccardIndex;
//     //const mxArray* spm1;
//     //double* spm1Features;
//     const mxArray* largerBoxIds1;
//     //unsigned int box2Id;
//     double saliency2;
//     const mxArray* detectionClass2;
//     size_t buflen2;
//     string detectionClass2String;
//     unsigned int stringIterator2;
//     double detectionClass2JaccardIndex;
//     //const mxArray* spm2;
//     //double* spm2Features;
//     //const mwSize* spmDimensions;
//     //unsigned int featureDimension;
//     //double spmDistance;
//     //unsigned int fDim;
//     //double spmSimilarity;
//     double contrastHOG;
//     double contrastSPM;
// //     double skipSPM1;
// //     double skipSPM2;
//     
//     omp_set_num_threads(4);
// 
//   
//     auto start = system_clock::now();
//     #pragma omp parallel for default(none) \
//                 shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, /*spmSimilarityMatrixPointer,*/ similarityMatrix, saliencyFieldId, /*skipSPMFieldId,*/ \
//                         detectionClassFieldId, detectionClassJaccardIndexFieldId, /*SPMFieldId,*/ largerBoxIdsFieldId, prhs, cout, maxHOGMatchScores/*, maxSPMMatchScores*/) \
//                         private(box1Id, saliency1, detectionClass1, buflen1, detectionClass1String, stringIterator1, detectionClass1JaccardIndex, \
//                                 /*spm1, spm1Features,*/ largerBoxIds1, box2Id, saliency2, detectionClass2, buflen2, detectionClass2String, stringIterator2, \
//                                 detectionClass2JaccardIndex, /*spm2, spm2Features, spmDimensions, featureDimension, spmDistance, fDim, spmSimilarity,*/ contrastHOG/*, contrastSPM, skipSPM1, skipSPM2*/)                          
//     for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
//     {
//         
//         saliency1 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, saliencyFieldId) ) )[0];
//         
//         detectionClass1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassFieldId);        
// 
//         buflen1 = mxGetN(detectionClass1);
//         detectionClass1String = string(buflen1, ' ');
//         for (stringIterator1 = 0; stringIterator1 < buflen1; stringIterator1++)
//         {
//             detectionClass1String[stringIterator1] = ((char*)mxGetData(detectionClass1))[0];
//         }
//         
// 
//         detectionClass1JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
//         
//         //spm1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, SPMFieldId);
//         //spm1Features = (double*)mxGetData(spm1);
//         
//         largerBoxIds1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, largerBoxIdsFieldId);
//         
// //         skipSPM1 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, skipSPMFieldId) ) )[0];
//         
//                 
//         for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
//         {
//             
//             saliency2 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, saliencyFieldId) ) )[0];
//             
//             detectionClass2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassFieldId);
//             
//             buflen2 = mxGetN(detectionClass2);
//             detectionClass2String = string(buflen2, ' ');
//             for (stringIterator2 = 0; stringIterator2 < buflen2; stringIterator2++)
//             {
//                 detectionClass2String[stringIterator2] = ((char*)mxGetData(detectionClass2))[0];
//             }
//                         
//             detectionClass2JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
//             
// //             skipSPM2 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, skipSPMFieldId) ) )[0];
//             
//             //spm2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, SPMFieldId);
//             //spm2Features = (double*)mxGetData(spm2);
//                         
//             // add saliency scores
//             similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (saliency1 + saliency2);
//             
//             // add detection scores
//             if( !(detectionClass1String.compare("background") == 0 || detectionClass2String.compare("background") == 0) )
//             {
//                 if ( detectionClass1String.compare(detectionClass2String) == 0 )
//                 {
//                     similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (detectionClass1JaccardIndex + detectionClass2JaccardIndex);
//                 }
//             }
//             
//             // add similarity between SPMs
//             //spmDimensions = mxGetDimensions(spm1);
//             //featureDimension = (unsigned int)spmDimensions[1];
//             
// //             spmDistance = 0;
// //             for (fDim = 0; fDim < featureDimension; fDim++)
// //             {
// //                 spmDistance +=  ( (spm1Features[fDim] - spm2Features[fDim]) * (spm1Features[fDim] - spm2Features[fDim]) ) /
// //                         ( spm1Features[fDim] + spm2Features[fDim] + DBL_EPSILON);                
// //             }
// //             spmDistance  = 0.5 * spmDistance;
// //             spmSimilarity = 1.0 - spmDistance;
//             
// //             contrastSPM = spmSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
// //                     maxOuterContrast(spmSimilarityMatrixPointer, totalNumberOfBoxes1, totalNumberOfBoxes2, largerBoxIds1, maxSPMMatchScores);
// //             
// //             if (skipSPM1 == 1 || skipSPM2 == 1)
// //             {
// //                 contrastSPM = -1.0;
// //             }
//                         
//             //similarityMatrix.at(box1Id).at(box2Id) += contrastSPM;
//                         
//             // add similarity between HOGs
//             contrastHOG = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
//                     maxOuterContrast(hogSimilarityMatrixPointer, totalNumberOfBoxes1, totalNumberOfBoxes2, largerBoxIds1, maxHOGMatchScores);
//             
//             similarityMatrix.at(box1Id).at(box2Id) += contrastHOG;
//             
//         }
//         
//         //cout << "Box1Id: " << box1Id << " vs. all done..." << endl;
//     }
//     auto end = system_clock::now();
//     auto elapsed = duration_cast<milliseconds>(end-start);
//     
//     cout << "Time elapsed: " << elapsed.count() << " milliseconds." << endl;
//     
//     
//       
//     mwSize resultMatrixDims[2];
//     resultMatrixDims[0] = totalNumberOfBoxes1;
//     resultMatrixDims[1] = totalNumberOfBoxes2;
//     plhs[0] = mxCreateNumericArray(2, resultMatrixDims, mxDOUBLE_CLASS, mxREAL);
//     double* resultMatrixPointer = (double*)mxGetData(plhs[0]);
//     
//     for (unsigned int box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
//     {
//         for (unsigned int box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
//         {
//             resultMatrixPointer[totalNumberOfBoxes1 * box2Id + box1Id] = similarityMatrix.at(box1Id).at(box2Id);
//         }
//     }
    
    
}
