#include <cstdlib>
#include <vector>

#include <chrono>

#include "./matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {
    double time_sum = 0;

    int nx = 41;
    int ny = 41;
    int nt = 15000;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    std::vector<double> x(ny*nx);
    std::vector<double> y(ny*nx);
    for (int j=0; j<ny; ++j) {
        for (int i=0; i<nx; ++i) {
            x[j*nx+i] = dx*j;
            y[j*nx+i] = dy*i;
        }
    }

    double u[ny*nx] = {0};
    double v[ny*nx] = {0};
    double p[ny*nx] = {0};
    double b[ny*nx] = {0};
    
    for (int n=0; n<nt; ++n) {
        auto start = std::chrono::system_clock::now();

        for (int j=1; j<ny-1; ++j) {
            for (int i=1; i<nx-1; ++i) {
                b[j*ny+i] = rho * (
                    1 / dt * ((u[j*ny+(i+1)] - u[j*ny+(i-1)]) / (2 * dx) + (v[(j+1)*ny+i] - v[(j-1)*ny+i]) / (2 * dy))
                    - std::pow((u[j*ny+(i+1)] - u[j*ny+(i-1)]) / (2 * dx), 2)
                    - 2 * ((u[(j+1)*ny+i] - u[(j-1)*ny+i]) / (2 * dy) * (v[j*ny+(i+1)] - v[j*ny+(i-1)]) / (2 * dx))
                    - std::pow((v[(j+1)*ny+i] - v[(j-1)*ny+i]) / (2 * dy), 2)
                );
            }
        }

        for (int it=0; it<nit; ++it) {
            double pn[ny*ny+nx];
            memcpy(pn, p, sizeof(p));
            for (int j=1; j<ny-1; ++j) {
                for (int i=1; i<nx-1; ++i) {
                    p[j*ny+i] = (
                        std::pow(dy, 2) * (pn[j*ny+(i+1)] + pn[j*ny+(i-1)])
                        + std::pow(dx, 2) * (pn[(j+1)*ny+i] + pn[(j-1)*ny+i])
                        - b[j*ny+i] * std::pow(dx, 2) * std::pow(dy, 2)
                    ) / (2 * (std::pow(dx, 2) + std::pow(dy, 2)));
                }
                for (int i=0; i<ny; ++i) p[i*ny+(nx-1)] = p[i*ny+nx-2];
                for (int i=0; i<nx; ++i) p[0*ny+i] = p[1*ny+i];
                for (int i=0; i<ny; ++i) p[i*ny+0] = p[i*ny+1];
                for (int i=0; i<nx; ++i) p[(ny-1)*ny+i] = 0;
            }
        }

        double un[ny*ny+nx], vn[ny*ny+nx];
        memcpy(un, u, sizeof(u));
        memcpy(vn, v, sizeof(v));
        for (int j=1; j<ny-1; ++j) {
            for (int i=1; i<nx-1; ++i) {
                u[j*ny+i] = un[j*ny+i] - un[j*ny+i] * dt / dx * (un[j*ny+i] - un[j*ny+(i-1)])
                                       - un[j*ny+i] * dt / dy * (un[j*ny+i] - un[(j-1)*ny+i])
                                       - dt / (2 * rho * dx) * (p[j*ny+(i+1)] - p[j*ny+(i-1)])
                                       + nu * dt / std::pow(dx, 2) * (un[j*ny+(i+1)] - 2 * un[j*ny+i] + un[j*ny+(i-1)])
                                       + nu * dt / std::pow(dy, 2) * (un[(j+1)*ny+i] - 2 * un[j*ny+i] + un[(j-1)*ny+i]);
                v[j*ny+i] = vn[j*ny+i] - vn[j*ny+i] * dt / dx * (vn[j*ny+i] - vn[j*ny+(i-1)])
                                       - vn[j*ny+i] * dt / dy * (vn[j*ny+i] - vn[(j-1)*ny+i])
                                       - dt / (2 * rho * dx) * (p[(j+1)*ny+i] - p[(j-1)*ny+i])
                                       + nu * dt / std::pow(dx, 2) * (vn[j*ny+(i+1)] - 2 * vn[j*ny+i] + vn[j*ny+(i-1)])
                                       + nu * dt / std::pow(dy, 2) * (vn[(j+1)*ny+i] - 2 * vn[j*ny+i] + vn[(j-1)*ny+i]);
            }
        }
        for (int i=0; i<nx; ++i) {
            u[0*ny+i] = 0;
            u[(ny-1)*ny+i] = 1;
            v[0*ny+i] = 0;
            v[(ny-1)*ny+i] = 0;
        }
        for (int i=0; i<ny; ++i) {
            u[i*ny+0] = 0;
            u[i*ny+(nx-1)] = 0;
            v[i*ny+0] = 0;
            v[i*ny+(nx-1)] = 0;
        }
        
        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        time_sum += time;
        if (n%50 == 0) {
            std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            printf("timestep: %5d; %s", n, std::ctime(&now));
            printf("time: %.2f [msec]\n", time_sum/50);
            time_sum = 0;
        }
        
        // plt::contourf(X, Y, p, alpha=0.5, cmap=plt::cm.coolwarm);
        std::vector<double> u_vec(u, u + sizeof(u)/sizeof(u[0])), v_vec(v, v + sizeof(v)/sizeof(v[0]));
        plt::quiver(y, x, u_vec, v_vec);
        plt::pause(.01);
        plt::clf();
    }
    plt::show();
} 
