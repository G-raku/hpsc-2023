#include <cstdio>
#include <cstdlib>
#include <vector>

#include <chrono>

__global__ void calc_b(double *u, double *v, double *p, double *b, double dx, double dy, double dt, double rho, double nu) {
    int j = blockIdx.x;
    int i = threadIdx.x;
    int ny = blockDim.x;

    if (j == 0 || j == ny-1 || i == 0 || i == ny-1) return;
    b[j*ny+i] = rho * (
        1 / dt * ((u[j*ny+(i+1)] - u[j*ny+(i-1)]) / (2 * dx) + (v[(j+1)*ny+i] - v[(j-1)*ny+i]) / (2 * dy))
        - ((u[j*ny+(i+1)] - u[j*ny+(i-1)]) / (2 * dx)) * ((u[j*ny+(i+1)] - u[j*ny+(i-1)]) / (2 * dx))
        - 2 * ((u[(j+1)*ny+i] - u[(j-1)*ny+i]) / (2 * dy) * (v[j*ny+(i+1)] - v[j*ny+(i-1)]) / (2 * dx))
        - ((v[(j+1)*ny+i] - v[(j-1)*ny+i]) / (2 * dy)) * ((v[(j+1)*ny+i] - v[(j-1)*ny+i]) / (2 * dy))
    );
}

__global__ void calc_p(double *u, double *v, double *p, double *b, double *pn, double dx, double dy, double dt, double rho, double nu) {
    int j = blockIdx.x;
    int i = threadIdx.x;
    int ny = blockDim.x;
    int nx = blockDim.x;

    p[j*ny+i] = (
        dy*dy * (pn[j*ny+(i+1)] + pn[j*ny+(i-1)])
        + dx*dx * (pn[(j+1)*ny+i] + pn[(j-1)*ny+i])
        - b[j*ny+i] * dx*dx * dy*dy
    ) / (2 * (dx*dx + std::pow(dy, 2)));
    __syncthreads();

    p[j*ny+(nx-1)] = p[j*ny+nx-2];
    p[0*ny+i] = p[1*ny+i];
    p[j*ny+0] = p[j*ny+1];
    p[(ny-1)*ny+i] = 0;
}

__global__ void calc_uv(double *u, double *v, double *p, double *b, double *un, double *vn, double dx, double dy, double dt, double rho, double nu) {
    int j = blockIdx.x;
    int i = threadIdx.x;
    int ny = blockDim.x;
    int nx = blockDim.x;

    u[j*ny+i] = un[j*ny+i]
                - un[j*ny+i] * dt / dx * (un[j*ny+i] - un[j*ny+(i-1)])
                - un[j*ny+i] * dt / dy * (un[j*ny+i] - un[(j-1)*ny+i])
                - dt / (2 * rho * dx) * (p[j*ny+(i+1)] - p[j*ny+(i-1)])
                + nu * dt / (dx*dx) * (un[j*ny+(i+1)] - 2 * un[j*ny+i] + un[j*ny+(i-1)])
                + nu * dt / (dy*dy) * (un[(j+1)*ny+i] - 2 * un[j*ny+i] + un[(j-1)*ny+i]);
    v[j*ny+i] = vn[j*ny+i] 
                - vn[j*ny+i] * dt / dx * (vn[j*ny+i] - vn[j*ny+(i-1)])
                - vn[j*ny+i] * dt / dy * (vn[j*ny+i] - vn[(j-1)*ny+i])
                - dt / (2 * rho * dx) * (p[(j+1)*ny+i] - p[(j-1)*ny+i])
                + nu * dt / (dx*dx) * (vn[j*ny+(i+1)] - 2 * vn[j*ny+i] + vn[j*ny+(i-1)])
                + nu * dt / (dy*dy) * (vn[(j+1)*ny+i] - 2 * vn[j*ny+i] + vn[(j-1)*ny+i]);
    __syncthreads();

    u[0*ny+i] = 0;
    u[i*ny+0] = 0;
    u[i*ny+(nx-1)] = 0;
    u[(ny-1)*ny+i] = 1;

    v[0*ny+i] = 0;
    v[i*ny+0] = 0;
    v[i*ny+(nx-1)] = 0;
    v[(ny-1)*ny+i] = 0;
}

int main() {
    double time_sum = 0;

    int nx = 41;
    int ny = 41;
    int nt = 10000;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    double *d_u, *d_v, *d_p, *d_b;
    double *u  , *v  , *p  , *b  ;
    cudaMalloc((void **)&d_u, ny*nx*sizeof(double));
    cudaMalloc((void **)&d_v, ny*nx*sizeof(double));
    cudaMalloc((void **)&d_p, ny*nx*sizeof(double));
    cudaMalloc((void **)&d_b, ny*nx*sizeof(double));
    u = (double*)calloc(ny*nx, sizeof(double));
    v = (double*)calloc(ny*nx, sizeof(double));
    p = (double*)calloc(ny*nx, sizeof(double));
    b = (double*)calloc(ny*nx, sizeof(double));

    double *d_pn, *d_un, *d_vn;
    cudaMalloc((void **)&d_pn, ny*nx*sizeof(double));
    cudaMalloc((void **)&d_un, ny*nx*sizeof(double));
    cudaMalloc((void **)&d_vn, ny*nx*sizeof(double));

    for (int n=0; n<nt; ++n) {
        auto start = std::chrono::system_clock::now();

        cudaMemcpy(d_u, u, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p, p, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        calc_b<<<ny, nx>>>(d_u, d_v, d_p, d_b, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();
        cudaMemcpy(b, d_b, ny*nx*sizeof(double), cudaMemcpyDeviceToHost);
        // for (int j=0; j<ny; j++) {
        //     for (int i=0; i<nx; ++i) {
        //         printf("%.2f ", v[j*ny+i]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        // if (n==10) break;

        for (int it=0; it<nit; ++it) {
            cudaMemcpy(d_p, p, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pn, p, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
            calc_p<<<ny, nx>>>(d_u, d_v, d_p, d_b, d_pn, dx, dy, dt, rho, nu);
            cudaDeviceSynchronize();
            cudaMemcpy(p, d_p, ny*nx*sizeof(double), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(d_u, u, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_p, p, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_un, u, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vn, v, ny*nx*sizeof(double), cudaMemcpyHostToDevice);
        calc_uv<<<ny, nx>>>(d_u, d_v, d_p, d_b, d_un, d_vn, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();
        cudaMemcpy(u, d_u, ny*nx*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, d_v, ny*nx*sizeof(double), cudaMemcpyDeviceToHost);
        
        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        time_sum += time;
        if (n%1000 == 0) {
            std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            printf("timestep: %5d; %s", n, std::ctime(&now));
            printf("time: %.3f [ms]\n", time_sum/1000/1000);
            time_sum = 0;
        }
    }
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_pn);
    cudaFree(d_un);
    cudaFree(d_vn);
    free(u);
    free(v);
    free(p);
    free(b);
} 
