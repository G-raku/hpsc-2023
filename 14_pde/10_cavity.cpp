#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <omp.h>

#include "./matplotlibcpp.h"
namespace plt = matplotlibcpp;

typedef std::vector<std::vector<double>> matrix;

int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = .01;
    double rho = 1.;
    double nu = .02;

    std::vector<double> x(ny*nx);
    std::vector<double> y(ny*nx);
    matrix X(ny, std::vector<double>(nx));
    matrix Y(ny, std::vector<double>(nx));
    for (int j=0; j<ny; ++j) {
        for (int i=0; i<nx; ++i) {
            x[j*ny+i] = X[j][i] = dx*j;
            y[j*ny+i] = Y[j][i] = dy*i;
        }
    }

    matrix u(ny, std::vector<double>(nx));
    matrix v(ny, std::vector<double>(nx));
    matrix p(ny, std::vector<double>(nx));
    matrix b(ny, std::vector<double>(nx));
    
    for (int n=0; n<nt; ++n) {
        for (int j=1; j<ny-1; ++j) {
            for (int i=1; i<nx-1; ++i) {
                b[j][i] = rho * (
                    1 / dt
                    * ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy))
                    - std::pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2)
                    - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx))
                    - std::pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2)
                );
            }
        }

        for (int it=0; it<nit; ++it) {
            matrix pn(ny, std::vector<double>(nx));
            for (int i=0; i<ny; ++i) std::copy(p[i].begin(), p[i].end(), pn[i].begin());
            for (int j=1; j<ny-1; ++j) {
                for (int i=1; i<nx-1; ++i) {
                    p[j][i] = (
                        std::pow(dy, 2) * (pn[j][i+1] + pn[j][i-1])
                        + std::pow(dx, 2) * (pn[j+1][i] + pn[j-1][i])
                        - b[j][i] * std::pow(dx, 2) * std::pow(dy, 2)
                    ) / (2 * (std::pow(dx, 2) + std::pow(dy, 2)));
                }
                for (int i=0; i<ny; ++i) p[i][p[0].size()-1] = p[i][p[0].size()-2];
                for (int i=0; i<nx; ++i) p[0][i] = p[1][i];
                for (int i=0; i<ny; ++i) p[i][0] = p[i][1];
                for (int i=0; i<nx; ++i) p[p.size()-1][i] = 0;
            }
        }

        matrix un(ny, std::vector<double>(nx));
        for (int i=0; i<ny; ++i) std::copy(u[i].begin(), u[i].end(), un[i].begin());
        matrix vn(ny, std::vector<double>(nx));
        for (int i=0; i<ny; ++i) std::copy(v[i].begin(), v[i].end(), vn[i].begin());
        for (int j=1; j<ny-1; ++j) {
            for (int i=1; i<nx-1; ++i) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])
                                   - un[j][i] * dt / dy * (un[j][i] - un[j-1][i])
                                   - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                                   + nu * dt / std::pow(dx, 2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                                   + nu * dt / std::pow(dy, 2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i-1])
                                   - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i])
                                   - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                                   + nu * dt / std::pow(dx, 2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                                   + nu * dt / std::pow(dy, 2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }
        for (int i=0; i<nx; ++i) u[0][i] = 0;
        for (int i=0; i<ny; ++i) u[i][0] = 0;
        for (int i=0; i<nx; ++i) u[u.size()-1][i] = 1;
        for (int i=0; i<ny; ++i) u[i][u[0].size()-1] = 0;
        for (int i=0; i<nx; ++i) v[0][i] = 0;
        for (int i=0; i<ny; ++i) v[i][0] = 0;
        for (int i=0; i<nx; ++i) v[v.size()-1][i] = 0;
        for (int i=0; i<ny; ++i) v[i][v[0].size()-1] = 0;
        
        // plt::contourf(X, Y, p, alpha=0.5, cmap=plt::cm.coolwarm);
        plt::quiver(X[20], Y[20], u[20], v[20]);
        plt::pause(.01);
        plt::clf();
    }
    plt::show();
} 
