#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#pragma omp declare reduction(vec_int_plus : std::vector<int> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

std::vector<int> scan(std::vector<int>);

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
#pragma omp parallel for reduction(vec_int_plus:bucket)
  for (int i=0; i<n; i++)
    bucket[key[i]]++;

  // std::vector<int> offset(range,0);
  // for (int i=1; i<range; i++) 
  //   offset[i] = offset[i-1] + bucket[i-1];
  std::vector<int> offset = scan(bucket);
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    offset[i] -= bucket[i];
  }
  
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}

std::vector<int> scan(std::vector<int> vec) {
  std::vector<int> tmp(vec.size());
#pragma omp parallel
  for (int j=1; j<vec.size(); j<<=1) {
#pragma omp for
    for (int i=0; i<vec.size(); i++) {
      tmp[i] = vec[i];
    }
#pragma omp for
    for (int i=j; i<vec.size(); i++) {
      vec[i] += tmp[i-j];
    }
  }
  return vec;
}