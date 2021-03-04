#include <iostream>
#include <fstream>

#include <Eigen/Sparse>

const size_t NX = 50;
const size_t NY = 50;
const size_t N = NX * NY;

const double dx = 0.05;
const double dt = 0.001;
const double r = dx / dt;
const double rho = 1.0;
const double nu  = 10.0;
const double u_lid = 1.0;
const double p0 = 0.0;

using Vector  = Eigen::VectorXd;
using Matrix  = Eigen::SparseMatrix<double>;
using View    = Eigen::Map<Eigen::Matrix<double, NX, NY>>;
using Triplet =  Eigen::Triplet<double>;

double access(View& v, double l, double r, double b, double t, int i, int j) {
  if (i == -1)
    return l;
  else if (i == NX)
    return r;
  else if (j == -1)
    return b;
  else if (j == NY)
    return t;
  else
    return v(i,j);
}

void assign(Matrix& A, int fo, int xo, int fi, int fj, int xi, int xj, double v) {
  if (0 <= fi && fi < NX &&
      0 <= fj && fj < NY &&
      0 <= xi && xi < NX &&
      0 <= xj && xj < NY) {
    size_t f_idx = fo + fi + NX*fj;
    size_t x_idx = xo + xi + NX*xj;
    A.coeffRef(f_idx, x_idx) = v;
  }
}

void calc(
    Vector& x_prev,
    Vector& x,
    Vector& y,
    Matrix& J) {

  View ux_prev_view(x_prev.data());
  View uy_prev_view(x_prev.data() + N);
  View p_prev_view(x_prev.data() + 2*N);

  View ux_view(x.data());
  View uy_view(x.data() + N);
  View p_view(x.data() + 2*N);

  View f1(y.data());
  View f2(y.data() + N);
  View f3(y.data() + 2*N);

  auto ux_prev = [&] (int i, int j) {return access(ux_prev_view, 0.0, 0.0, 0.0, u_lid, i, j);};
  auto uy_prev = [&] (int i, int j) {return access(uy_prev_view, 0.0, 0.0, 0.0, 0.0, i, j);};
  auto p_prev  = [&] (int i, int j) {return access(p_prev_view, p0, p0, p0, p0, i, j);};

  auto ux = [&] (int i, int j) {return access(ux_view, 0.0, 0.0, 0.0, u_lid, i, j);};
  auto uy = [&] (int i, int j) {return access(uy_view, 0.0, 0.0, 0.0, 0.0, i, j);};
  auto p  = [&] (int i, int j) {return access(p_view, p0, p0, p0, p0, i, j);};

  auto assign_f1_ux = [&] (int fi, int fj, int xi, int xj, double v) {assign(J,   0,   0, fi, fj, xi, xj, v);};
  auto assign_f1_uy = [&] (int fi, int fj, int xi, int xj, double v) {assign(J,   0,   N, fi, fj, xi, xj, v);};
  auto assign_f1_p  = [&] (int fi, int fj, int xi, int xj, double v) {assign(J,   0, 2*N, fi, fj, xi, xj, v);};

  auto assign_f2_ux = [&] (int fi, int fj, int xi, int xj, double v) {assign(J,   N,   0, fi, fj, xi, xj, v);};
  auto assign_f2_uy = [&] (int fi, int fj, int xi, int xj, double v) {assign(J,   N,   N, fi, fj, xi, xj, v);};
  auto assign_f2_p  = [&] (int fi, int fj, int xi, int xj, double v) {assign(J,   N, 2*N, fi, fj, xi, xj, v);};

  auto assign_f3_ux = [&] (int fi, int fj, int xi, int xj, double v) {assign(J, 2*N,   0, fi, fj, xi, xj, v);};
  auto assign_f3_uy = [&] (int fi, int fj, int xi, int xj, double v) {assign(J, 2*N,   N, fi, fj, xi, xj, v);};
  auto assign_f3_p  = [&] (int fi, int fj, int xi, int xj, double v) {assign(J, 2*N, 2*N, fi, fj, xi, xj, v);};

  /* Boundary condition
   *  ux(-1,j) = 0.0
   *  ux(NX,j) = 0.0
   *  ux(i,-1) = 0.0
   *  ux(i,NY) = u_lid
   *
   *  uy(-1,j) = 0.0
   *  uy(NX,j) = 0.0
   *  uy(i,-1) = 0.0
   *  uy(i,NY) = 0.0
   *
   *  p(-1,j) = p0
   *  p(NX,j) = p0
   *  p(i,-1) = p0
   *  p(i,NY) = p0
   */
 
  for (int j=0; j<NY; j++) {
    for (int i=0; i<NX; i++) {
      const double ux_ = ux(i,j) + ux_prev(i,j);
      const double uy_ = uy(i,j) + uy_prev(i,j);
      const double ux_t_ = ux(i,j) - ux_prev(i,j);
      const double ux_x_ = ux(i+1,j) - ux(i-1,j) + ux_prev(i+1,j) - ux_prev(i-1,j);
      const double ux_y_ = ux(i,j+1) - ux(i,j-1) + ux_prev(i,j+1) - ux_prev(i,j-1);
      const double p_x_ = p(i+1,j) - p(i-1,j) + p_prev(i+1,j) - p_prev(i-1,j);
      const double la_ux_ = ux(i+1,j) + ux(i-1,j) + ux(i,j+1) + ux(i,j-1) - 4.0*ux(i,j);
      f1(i,j) = 8.0 * r * ux_t_
              + ux_ * ux_x_
              + uy_ * ux_y_
              + (2.0 / rho) * p_x_
              - (8.0 * nu / dx) * la_ux_;
      assign_f1_ux(i, j, i,   j, 8.0*r + ux_x_ + (32.0 * nu / dx));
      assign_f1_ux(i, j, i+1, j,           ux_ - (8.0 * nu / dx));
      assign_f1_ux(i, j, i-1, j,          -ux_ - (8.0 * nu / dx));
      assign_f1_ux(i, j, i,   j+1,         uy_ - (8.0 * nu / dx));
      assign_f1_ux(i, j, i,   j-1,        -uy_ - (8.0 * nu / dx));
      assign_f1_uy(i, j, i,   j,         ux_y_);
      assign_f1_p (i, j, i+1, j,       2.0/rho);
      assign_f1_p (i, j, i-1, j,      -2.0/rho);
    }
  }

  for (int j=0; j<NY; j++) {
    for (int i=0; i<NX; i++) {
      const double ux_ = ux(i,j) + ux_prev(i,j);
      const double uy_ = uy(i,j) + uy_prev(i,j);
      const double uy_t_ = uy(i,j) - uy_prev(i,j);
      const double uy_x_ = uy(i+1,j) - uy(i-1,j) + uy_prev(i+1,j) - uy_prev(i-1,j);
      const double uy_y_ = uy(i,j+1) - uy(i,j-1) + uy_prev(i,j+1) - uy_prev(i,j-1);
      const double p_y_ = p(i,j+1) - p(i,j-1) + p_prev(i,j+1) - p_prev(i,j-1);
      const double la_uy_ = uy(i+1,j) + uy(i-1,j) + uy(i,j+1) + uy(i,j-1) - 4.0*uy(i,j);
      f2(i,j) = 8.0 * r * uy_t_
              + ux_ * uy_x_
              + uy_ * uy_y_
              + (2.0 / rho) * p_y_
              - (8.0 * nu / dx) * la_uy_;
      assign_f2_uy(i, j, i,   j, 8.0*r + uy_y_ + (32.0 * nu / dx));
      assign_f2_uy(i, j, i+1, j,           ux_ - (8.0 * nu / dx));
      assign_f2_uy(i, j, i-1, j,          -ux_ - (8.0 * nu / dx));
      assign_f2_uy(i, j, i,   j+1,         uy_ - (8.0 * nu / dx));
      assign_f2_uy(i, j, i,   j-1,        -uy_ - (8.0 * nu / dx));
      assign_f2_ux(i, j, i,   j,         uy_x_);
      assign_f2_p (i, j, i,   j+1,     2.0/rho);
      assign_f2_p (i, j, i,   j-1,    -2.0/rho);
    }
  }

  for (int j=0; j<NY; j++) {
    for (int i=0; i<NX; i++) {
      const double ux_x_ = ux(i+1,j) - ux(i-1,j);
      const double ux_y_ = ux(i,j+1) - ux(i,j-1);
      const double uy_x_ = uy(i+1,j) - uy(i-1,j);
      const double uy_y_ = uy(i,j+1) - uy(i,j-1);
      const double la_p_ = p(i+1,j) + p(i-1,j) + p(i,j+1) + p(i,j-1) - 4.0*p(i,j);
      f3(i,j) = ux_x_*ux_x_ + 2.0*uy_x_*ux_y_ + uy_y_*uy_y_ + (4.0/rho) * la_p_;
      assign_f3_ux(i, j, i+1, j,    2.0*ux_x_);
      assign_f3_ux(i, j, i-1, j,   -2.0*ux_x_);
      assign_f3_ux(i, j, i,   j+1,  2.0*uy_x_);
      assign_f3_ux(i, j, i,   j-1, -2.0*uy_x_);
      assign_f3_uy(i, j, i+1, j,    2.0*ux_y_);
      assign_f3_uy(i, j, i-1, j,   -2.0*ux_y_);
      assign_f3_uy(i, j, i,   j+1,  2.0*uy_y_);
      assign_f3_uy(i, j, i,   j-1, -2.0*uy_y_);
      assign_f3_p (i, j, i,   j,    -16.0/rho);
      assign_f3_p (i, j, i+1, j,      4.0/rho);
      assign_f3_p (i, j, i-1, j,      4.0/rho);
      assign_f3_p (i, j, i,   j+1,    4.0/rho);
      assign_f3_p (i, j, i,   j-1,    4.0/rho);
    }
  }

  return;
}

int main() {
  Vector x_prev(3*N);
  Vector x(3*N);
  Vector y(3*N);
  Matrix J(3*N, 3*N);
  J.reserve(20*N);

  View ux_view(x.data());
  View uy_view(x.data() + N);
  View p_view(x.data() + 2*N);

  for (size_t i=0; i<2*N; i++)
    x_prev[i] = 0.0;
  for (size_t i=2*N; i<3*N; i++)
    x_prev[i] = p0;

  x = x_prev;

  Eigen::BiCGSTAB<Matrix> solver;
  Eigen::MatrixXf ux_;
  Eigen::MatrixXf uy_;
  Eigen::MatrixXf p_;

  std::ofstream file("result", std::ios::out | std::ios::binary);

  auto step = [&] () {
    calc(x_prev, x, y, J);
    solver.compute(J);
    x -= solver.solve(y);

    calc(x_prev, x, y, J);
    solver.compute(J);
    x -= solver.solve(y);

    x_prev = x;

    file.write((char *)x.data(), sizeof(double)*3*N);
  };

  for (int i=0; i<50; i++)
    step();
  
  file.close();

  return 0;
}
