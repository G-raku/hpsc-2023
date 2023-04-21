#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 x_vec = _mm256_load_ps(x);
  __m256 y_vec = _mm256_load_ps(y);
  __m256 fx_vec = _mm256_load_ps(fx);
  __m256 fy_vec = _mm256_load_ps(fy);
  for(int i=0; i<N; i++) {
    __m256 rx_vec = _mm256_sub_ps(x_vec, _mm256_set1_ps(x[i])); // rx[] = x[] - x[i], Distance between each point (x[]) and the i-th point (x[i])
    __m256 ry_vec = _mm256_sub_ps(y_vec, _mm256_set1_ps(y[i])); // ry[] = y[] - y[i]
    __m256 one_rth_vec = _mm256_rsqrt_ps( // 1/r = 1/sqrt(rx^2 + ry^2)
      _mm256_add_ps(
        _mm256_mul_ps(rx_vec, rx_vec), _mm256_mul_ps(ry_vec, ry_vec)
      )
    );
    __m256 one_rth_vec3 = _mm256_mul_ps(_mm256_mul_ps(one_rth_vec, one_rth_vec), one_rth_vec); // (1/r)^3

    fx_vec = _mm256_sub_ps( // fx[] -= rx[] * m[i] * (1/r)^3
      fx_vec,
      _mm256_blend_ps(
        _mm256_mul_ps(_mm256_mul_ps(rx_vec, _mm256_set1_ps(m[i])), one_rth_vec3),
        _mm256_setzero_ps(),
        0b00000001<<i // mask i-th (0b..1..)
      )
    );
    fy_vec = _mm256_sub_ps( // fy[] -= ry[] * m[i] * (1/r)^3
      fy_vec,
      _mm256_blend_ps(
        _mm256_mul_ps(_mm256_mul_ps(ry_vec, _mm256_set1_ps(m[i])), one_rth_vec3),
        _mm256_setzero_ps(),
        0b00000001<<i // mask i-th (0b..1..)
      )
    );
  }
  _mm256_store_ps(fx, fx_vec);
  _mm256_store_ps(fy, fy_vec);
  for (int i=0; i<N; i++) {
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
